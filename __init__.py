import torch
import folder_paths
import comfy.sd
import comfy.model_management

import os

import folder_paths

import importlib

import numpy as np

# Importar el módulo desde custom_nodes/x-flux-comfyui/xflux/src/flux/util.py
flux_xlabs_util = importlib.import_module('custom_nodes.x-flux-comfyui.xflux.src.flux.util')

# Acceder a las funciones del módulo importado
load_controlnet = getattr(flux_xlabs_util, 'load_controlnet')


# Importar el módulo desde custom_nodes/x-flux-comfyui/xflux/src/flux/util.py
flux__xlabs_nodes = importlib.import_module('custom_nodes.x-flux-comfyui.nodes')

# Acceder a las funciones del módulo importado
load_checkpoint_controlnet = getattr(flux__xlabs_nodes, 'load_checkpoint_controlnet')

XlabsSampler = getattr(flux__xlabs_nodes, 'XlabsSampler')

current_device = "cuda:0"


dir_xlabs = os.path.join(folder_paths.models_dir, "xlabs")
os.makedirs(dir_xlabs, exist_ok=True)
dir_xlabs_controlnets = os.path.join(dir_xlabs, "controlnets")
os.makedirs(dir_xlabs_controlnets, exist_ok=True)

def get_torch_device_patched():
    global current_device
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
    ):
        return torch.device("cpu")

    return torch.device(current_device)


comfy.model_management.get_torch_device = get_torch_device_patched


class CheckpointLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, device):
        global current_device
        current_device = device

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


class UNETLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype, device):
        global current_device
        current_device = device

        dtype = None
        if weight_dtype == "fp8_e4m3fn":
            dtype = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            dtype = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("unet", unet_name)
        #model = comfy.sd.load_unet(unet_path, dtype=dtype)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"dtype":dtype})
        return (model,)


class VAELoaderMultiGPU:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(
            filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes)
        )
        decoder = next(
            filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes)
        )

        enc = comfy.utils.load_torch_file(
            folder_paths.get_full_path("vae_approx", encoder)
        )
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(
            folder_paths.get_full_path("vae_approx", decoder)
        )
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (s.vae_list(),),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    # TODO: scale factor?
    def load_vae(self, vae_name, device):
        global current_device
        current_device = device

        if vae_name in ["taesd", "taesdxl", "taesd3"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return (vae,)


class ControlNetLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name, device):
        global current_device
        current_device = device

        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return (controlnet,)


class CLIPLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "type": (
                    ["stable_diffusion", "stable_cascade", "sd3", "stable_audio"],
                ),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, device, type="stable_diffusion"):
        global current_device
        current_device = device

        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        return (clip,)


class DualCLIPLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "type": (["sdxl", "sd3", "flux"],),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type, device):
        global current_device
        current_device = device

        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        return (clip,)



class LoadFluxControlNetMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (["flux-dev", "flux-dev-fp8", "flux-schnell"],),
                            "controlnet_path": (folder_paths.get_filename_list("xlabs_controlnets"), ),
                            "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
        }}

    RETURN_TYPES = ("FluxControlNet",)
    RETURN_NAMES = ("ControlNet",)
    FUNCTION = "loadmodel"
    CATEGORY = "XLabsNodes"

    def loadmodel(self, model_name, controlnet_path, device):

        global current_device
        current_device = device

        controlnet = load_controlnet(model_name, device)
        checkpoint = load_checkpoint_controlnet(os.path.join(dir_xlabs_controlnets, controlnet_path))
        if checkpoint is not None:
            controlnet.load_state_dict(checkpoint)
            control_type = "canny"
        ret_controlnet = {
            "model": controlnet,
            "control_type": control_type,
        }

        return (ret_controlnet,)
    
class ApplyFluxControlNetMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "controlnet": ("FluxControlNet",),
                    "image": ("IMAGE", ),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
                },
                "optional": {
                    "controlnet_condition": ("ControlNetCondition", {"default": None}),
                }
        }

    RETURN_TYPES = ("ControlNetCondition",)
    RETURN_NAMES = ("controlnet_condition",)
    FUNCTION = "prepare"
    CATEGORY = "XLabsNodes"

    def prepare(self, controlnet, image, strength, device, controlnet_condition = None):
        global current_device
        current_device = device

        controlnet_image = torch.from_numpy((np.array(image) * 2) - 1)
        controlnet_image = controlnet_image.permute(0, 3, 1, 2).to(torch.bfloat16).to(device)

        if controlnet_condition is None:
            ret_cont = [{
                "img": controlnet_image,
                "controlnet_strength": strength,
                "model": controlnet["model"],
                "start": 0.0,
                "end": 1.0
            }]
        else:
            ret_cont = controlnet_condition+[{
                "img": controlnet_image,
                "controlnet_strength": strength,
                "model": controlnet["model"],
                "start": 0.0,
                "end": 1.0
            }]
        return (ret_cont,)

class XlabsSamplerMultiGPU(XlabsSampler):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "model": ("MODEL",),
                    "conditioning": ("CONDITIONING",),
                    "neg_conditioning": ("CONDITIONING",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT",  {"default": 20, "min": 1, "max": 100}),
                    "timestep_to_start_cfg": ("INT",  {"default": 20, "min": 0, "max": 100}),
                    "true_gs": ("FLOAT",  {"default": 3, "min": 0, "max": 100}),
                    "image_to_image_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
                },
            "optional": {
                    "latent_image": ("LATENT", {"default": None}),
                    "controlnet_condition": ("ControlNetCondition", {"default": None}),
                }
            } 

    def sampling(self, model, conditioning, neg_conditioning,
                 noise_seed, steps, timestep_to_start_cfg, true_gs,
                 image_to_image_strength, denoise_strength,
                 latent_image=None, controlnet_condition=None, device="cuda:0"
                 ):
        global current_device
        current_device = device

        
        torch.cuda.device(device)

        return super().sampling(model, conditioning, neg_conditioning,
                 noise_seed, steps, timestep_to_start_cfg, true_gs,
                 image_to_image_strength, denoise_strength,
                 latent_image, controlnet_condition)
        
NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderMultiGPU": CheckpointLoaderMultiGPU,
    "UNETLoaderMultiGPU": UNETLoaderMultiGPU,
    "VAELoaderMultiGPU": VAELoaderMultiGPU,
    "ControlNetLoaderMultiGPU": ControlNetLoaderMultiGPU,
    "CLIPLoaderMultiGPU": CLIPLoaderMultiGPU,
    "DualCLIPLoaderMultiGPU": DualCLIPLoaderMultiGPU,
    "LoadFluxControlNetMultiGPU": LoadFluxControlNetMultiGPU,
    "ApplyFluxControlNetMultiGPU": ApplyFluxControlNetMultiGPU,
    "XlabsSamplerMultiGPU": XlabsSamplerMultiGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderMultiGPU": "Load Checkpoint (Multi-GPU)",
    "UNETLoaderMultiGPU": "Load Diffusion Model (Multi-GPU)",
    "VAELoaderMultiGPU": "Load VAE (Multi-GPU)",
    "ControlNetLoaderMultiGPU": "Load ControlNet Model (Multi-GPU)",
    "CLIPLoaderMultiGPU": "Load CLIP (Multi-GPU)",
    "DualCLIPLoaderMultiGPU": "DualCLIPLoader (Multi-GPU)",
    "LoadFluxControlNetMultiGPU": "Load Flux ControlNet (Multi-GPU)",
    "ApplyFluxControlNetMultiGPU": "Apply Flux ControlNet (Multi-GPU)",
    "XlabsSamplerMultiGPU": "XlabsSampler (Multi-GPU)",
}
