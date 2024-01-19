import os
import sys
import time
import threading
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
import numpy as np

from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
import mss
import fire
import tkinter as tk

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline,ControlNetModel, StableDiffusionControlNetPipeline,StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from my_image_utils import pil2tensor
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

###############################################
# プロンプトはここ
###############################################
box_prompt = "1 girl"
###############################################


class StreamDiffusionControlNetSample(StreamDiffusion):
    def __init__(self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        ip_adapter = None):
        super().__init__(pipe,
            t_index_list,
            torch_dtype,
            width,
            height,
            do_add_noise,
            use_denoising_batch,
            frame_buffer_size,
            cfg_type,
            )
        self.ip_adapter=ip_adapter
        if pipe.controlnet != None:
            self.controlnet = pipe.controlnet
        self.input_latent = None
        self.ctl_image_t_buffer = None
        self.added_cond_kwargs=None

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
        ip_adapter_image = None
    ) -> None:
        self.do_classifier_free_guidance=False
        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta
        self.do_classifier_free_guidance = self.is_do_classifer_free_guicance()
        ##IPAdapterのため
        if self.ip_adapter:
            ip_adapter_image  = ip_adapter_image .resize((self.height, self.width))


            #SD IPADAPTERIMPL
            num_images_per_prompt = 1
            
            if ip_adapter_image is not None:
                image_embeds, negative_image_embeds = self.pipe.encode_image(ip_adapter_image, "cuda", num_images_per_prompt)


            print("image_embeded:{}".format(image_embeds.shape))
            
        
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None
  
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            #lora_scale=lora_scale,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        
        if self.ip_adapter:
            #IPADAPTER ORIGINAL IMPL
            image_embeds = image_embeds.repeat(self.batch_size, 1, 1)
            #image_embeds = image_embeds.repeat(4, 1, 1)
            if self.do_classifier_free_guidance:
                negative_image_embeds = negative_image_embeds.repeat(self.batch_size, 1, 1)
                
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
            
            #SD IPADAPTER IMPL
            self.added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
    
    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
        image = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
            #image = torch.concat([image, image], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent
        #print(image.shape)
        latent_model_input = x_t_latent_plus_uc #self.do_classifier_free_guidance
        controlnet_prompt_embeds = self.prompt_embeds
        control_model_input = latent_model_input
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t_list,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=1, #cond_scale,
                    guess_mode=False, #guess_mode,
                    return_dict=False,
                )

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            encoder_hidden_states=self.prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
            added_cond_kwargs=self.added_cond_kwargs
        )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred
    
    def is_do_classifer_free_guicance(self) :
        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True
        return do_classifier_free_guidance

    @torch.no_grad()
    def update_prompt(self, prompt: str, negative_prompt) -> None:
        do_classifier_free_guidance = self.is_do_classifer_free_guicance()

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)
    
        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )



    
    def predict_x0_batch(self, x_t_latent: torch.Tensor,
                         image = None) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer
        # todo とりあえず埋める。
        if self.ctl_image_t_buffer is None or self.x_t_latent_buffer.shape[0] >= self.ctl_image_t_buffer.shape[0]:
            self.ctl_image_t_buffer = image.repeat(self.x_t_latent_buffer.shape[0], 1, 1,1)
            
        prev_ctl_image_t_buffer = self.ctl_image_t_buffer 

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
                
                images = torch.cat(
                    (image, prev_ctl_image_t_buffer), dim=0
                )
                
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, image=images)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
                if self.cfg_type == "full":
                    self.ctl_image_t_buffer = images[:-2] # TODO 後ろ２つでいいのか？
                else:
                    self.ctl_image_t_buffer = images[:-1]
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx, image=image)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out


    @torch.no_grad()
    def ctlimg2img(self, batch_size: int = 1, ctlnet_image = None, keep_latent = False) -> torch.Tensor:
        if not keep_latent:
            self.input_latent = torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        else:
            if self.input_latent is None:
                latent = torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                    device=self.device, dtype=self.dtype
                )
                self.input_latent = latent
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tstart = time.time()
       
        #コントロールネット用の計算
        num_images_per_prompt = 1
        batch_size = 1
        guess_mode = False
        if self.pipe.controlnet != None:
            timage = self.pipe.prepare_image(
                    image=ctlnet_image,
                    width=self.width,
                    height=self.height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
            
        ctlnet_image = timage
        x_0_pred_out = self.predict_x0_batch(self.input_latent,ctlnet_image)
        tstart = time.time()
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        
        tstart = time.time()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

UPEER_FPS = 40
fps_interval = 1.0/UPEER_FPS
inputs = []
top = 0
left = 0




def screen(
    event: threading.Event(),
    height: int = 512,
    width: int = 512,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 512*2, "height": 512*2},
):
    global inputs
    
    with mss.mss() as sct:
        while True:
            if event.is_set():
                print("terminate read thread")
                break
            start_time = time.time()
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img.resize((height, width))
            inputs.append(pil2tensor(img))
            interval = time.time() - start_time
            fps_interval = 1.0/UPEER_FPS
            if interval < fps_interval:
                sleep_time = fps_interval - interval
                #print("screen:{}".format(sleep_time))
                
                time.sleep(sleep_time)

    print('exit : screen')
def dummy_screen(
        width: int,
        height: int,
):
    root = tk.Tk()
    root.title("Press Enter to start")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    root.attributes("-alpha", 0.8)
    root.configure(bg="black")
    def destroy(event):
        root.destroy()
    root.bind("<Return>", destroy)
    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()
    root.bind("<Configure>", update_geometry)
    root.mainloop()
    return {"top": top, "left": left, "width": width, "height": height}


def prompt_window(queue):
    def on_submit():
        global box_prompt
        entered_text = entry.get()
        print("入力されたテキスト:", entered_text)
        sv.set(entered_text)
        queue.put(entered_text)

    # メインウィンドウを作成
    root = tk.Tk()
    root.title("入力と送信")
    root.geometry(f"{600}x{300}")
    # ラベルを作成
    label = tk.Label(root, text="prompt")
    label.pack(pady=10)

    # 入力フィールドを作成
    de = box_prompt
    sv = tk.StringVar()
    entry = tk.Entry(root,textvariable=sv)
    entry.insert(0, de) 
    entry.pack(pady=10, fill=tk.X) 

    # 送信ボタンを作成
    submit_button = tk.Button(root, text="送信", command=on_submit)
    submit_button.pack(pady=10)

    # イベントループを開始
    root.mainloop()

def monitor_setting_process(
    width: int,
    height: int,
    monitor_sender: Connection,
) -> None:
    monitor = dummy_screen(width, height)
    monitor_sender.send(monitor)

base_img = None

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    monitor_receiver : Connection,
    prompt_queue
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """
    
    global inputs
    global box_prompt
    instep = 50
    ######################################################
    #パラメタ
    ######################################################
    adapter = True
    ip_adapter_image_filepath="../assets/img2img_example.png"
    
    t_index_list=[0, 16, 32, 45]
    #t_index_list=[0, 30, 45]
    #t_index_list=[0,10,20,30,40]
    cfg_type = "none"
    #cfg_type="full" #一応動くが・・・
    #RCFG系は非対応
    
    delta = 1.0

    #Trueで潜在空間の乱数を固定します。
    keep_latent=True

    # fullで有効
    negative_prompt = """(monochrome:0.8),(deformed:1.3),(malformed hands:1.4),(poorly drawn hands:1.4),(mutated fingers:1.4),(bad anatomy:1.3),(extra limbs:1.35),(poorly drawn face:1.4),(signature:1.2),(artist name:1.2),(watermark:1.2),(worst quality, low quality, normal quality:1.4), lowres,skin blemishes,extra fingers,fewer fingers,strange fingers,Hand grip,(lean),Strange eyes,(three arms),(Many arms),(watermarking)"""
    ######################################################


    # ControlNetモデルの準備
    controlnet_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    ).to("cuda")

    # ipAdapterのイメージエンコーダ
    image_encoder = None
    if adapter:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1",
        controlnet=controlnet_pose,
        image_encoder=image_encoder).to(
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

    if adapter:
        pipe.load_ip_adapter('h94/IP-Adapter', subfolder="models",
                              weight_name="ip-adapter_sd15.bin",
                              torch_dtype=torch.float16)
        pipe.set_ip_adapter_scale(1.0)

    
    # Diffusers pipelineをStreamDiffusionにラップ
    stream = StreamDiffusionControlNetSample(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
        cfg_type=cfg_type,
        width=width,
        height=height,
        ip_adapter = adapter
    )
    

    # 読み込んだモデルがLCMでなければマージする
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm") #Stable  Diffusion 1.5 のLCM LoRA
    pipe.set_adapters(["lcm"], adapter_weights=[1.0])
    
    # Tiny VAEで高速化
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    # xformersで高速化 ip adapter が効かなくなるので無効にする
    #pipe.enable_xformers_memory_efficient_attention()

    ip_adapter_image=None
    if adapter:
        print("prepare ip adapter")
        # 初期画像の準備
        ip_adapter_image  = load_image(ip_adapter_image_filepath)


    stream.prepare(
        prompt=box_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=instep,
        guidance_scale=guidance_scale,
        delta=delta,
        ip_adapter_image=ip_adapter_image
    )


    monitor = monitor_receiver.recv()

    event = threading.Event()
    event.clear()

    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))

    input_screen.start()
    time.sleep(1)
    current_prompt = box_prompt
    while True:
        try:
            if not close_queue.empty(): # closing check
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(fps_interval)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            new_prompt = current_prompt
            if not prompt_queue.empty():
                new_prompt = prompt_queue.get(block=False)
                
            prompt_change = False
            if current_prompt != new_prompt:  
                current_prompt = new_prompt  
            #    print("change",current_prompt,new_prompt)    

                stream.update_prompt(current_prompt, negative_prompt)
                prompt_change = True
                            
            input = input_batch.to(device=stream.device, dtype=stream.dtype)
            global base_img
            if base_img is None:
                base_img = input
                
            output_images = stream.ctlimg2img(ctlnet_image=input, keep_latent=keep_latent)

            
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                queue.put(output_image, block=False)
            process_time = time.time() - start_time
            if process_time <= fps_interval:
                time.sleep(fps_interval - process_time)
            process_time = time.time() - start_time
            fps = 1 / (process_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break

    print("closing image_generation_process...")
    event.set() # stop capture thread
    input_screen.join()
    print(f"fps: {fps}")

def main(
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "none",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    prompt_queue = ctx.Queue()
    close_queue = Queue()

    do_add_noise=False
    monitor_sender, monitor_receiver = ctx.Pipe()

    prompt_process = ctx.Process(
        target=prompt_window,
        args=(
            prompt_queue,
            ),
    )
    prompt_process.start()
    
    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            monitor_receiver,
            prompt_queue
            ),
    )
    process1.start()

    monitor_process = ctx.Process(
        target=monitor_setting_process,
        args=(
            width,
            height,
            monitor_sender,
            ),
    )
    monitor_process.start()
    monitor_process.join()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # terminate
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5) # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate() # force kill...
    process1.join()
    print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)