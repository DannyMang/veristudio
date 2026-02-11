"""
app_gradio.py — Gradio interactive driving demo with KV-cached causal generation.

Upload a starting dashcam image, enter a prompt, then use WASD buttons to
steer the vehicle. Each button press generates a video chunk using
WanI2VCausal (KV-cached, ~2-4s per chunk after the first).

Usage:
    python app_gradio.py \
        --ckpt_dir /path/to/lingbot-world-base-cam \
        --share  # optional: create a public Gradio link

Requirements:
    pip install gradio
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Ensure lingbot-world is on sys.path
_lingbot_root = Path(__file__).resolve().parent.parent.parent / 'lingbot-world'
if str(_lingbot_root) not in sys.path:
    sys.path.insert(0, str(_lingbot_root))

# Import driving pose generator from sibling script
from drive_interactive import DrivingPoseGenerator


def create_demo(args):
    """Build and return the Gradio Blocks app."""
    import gradio as gr
    from PIL import Image

    # Lazy-loaded model (only when first generation is requested)
    state = {
        'driver': None,
        'pose_gen': None,
        'current_image': None,
        'prompt': '',
        'chunk_idx': 0,
    }

    def _ensure_model():
        if state['driver'] is not None:
            return

        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS['i2v-A14B']
        logger.info(f"Loading WanI2VCausal from {args.ckpt_dir} ...")
        state['driver'] = wan.WanI2VCausal(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            t5_cpu=True,
            max_cache_chunks=args.max_cache_chunks,
            sampling_steps=args.steps,
            max_area=args.max_area,
        )
        logger.info("Model loaded.")

    def _ensure_pose_gen():
        if state['pose_gen'] is None:
            state['pose_gen'] = DrivingPoseGenerator(
                speed=args.speed, turn_rate=args.turn_rate,
                num_frames=args.frame_num)

    def _generate_chunk(keys_str):
        """Core generation: takes a WASD key string, returns (image, status)."""
        if state['current_image'] is None:
            return None, "Upload a starting image first."

        _ensure_model()
        _ensure_pose_gen()
        pg = state['pose_gen']

        keys = set(c for c in keys_str.lower() if c in 'wasd')
        poses, desc = pg.wasd_to_poses(keys)
        h, w = int(args.size.split('*')[0]), int(args.size.split('*')[1])
        intrinsics = pg.get_default_intrinsics(len(poses), width=w, height=h)

        t0 = time.time()
        video, last_frame = state['driver'].generate_chunk(
            img=state['current_image'],
            prompt=state['prompt'],
            c2ws=poses,
            intrinsics=intrinsics,
            frame_num=args.frame_num,
            shift=3.0 if '480' in args.size else 10.0,
            seed=args.seed + state['chunk_idx'],
        )
        dt = time.time() - t0

        if last_frame is not None:
            state['current_image'] = last_frame

        state['chunk_idx'] += 1
        yaw_deg = np.degrees(pg.yaw)
        status = (f"Chunk {state['chunk_idx']}: {desc} | "
                  f"{dt:.1f}s | Yaw: {yaw_deg:.0f} deg")
        return state['current_image'], status

    def on_upload(img, prompt):
        """Handle image upload + prompt entry."""
        if img is None:
            return None, "No image uploaded."
        state['current_image'] = img
        state['prompt'] = prompt or (
            "Dashcam footage of a vehicle driving on a road during daytime. "
            "Clear weather, suburban environment.")
        state['chunk_idx'] = 0
        # Reset pose generator and causal cache
        if state['pose_gen'] is not None:
            state['pose_gen'].reset()
        if state['driver'] is not None:
            state['driver'].reset()
        return img, "Ready. Use WASD buttons to drive."

    def on_forward():
        return _generate_chunk('w')

    def on_left():
        return _generate_chunk('wa')

    def on_right():
        return _generate_chunk('wd')

    def on_reverse():
        return _generate_chunk('s')

    def on_forward_left():
        return _generate_chunk('wa')

    def on_forward_right():
        return _generate_chunk('wd')

    def on_coast():
        return _generate_chunk('')

    def on_reset():
        if state['pose_gen'] is not None:
            state['pose_gen'].reset()
        if state['driver'] is not None:
            state['driver'].reset()
        state['chunk_idx'] = 0
        return state['current_image'], "Reset to origin."

    # Build UI
    with gr.Blocks(title="LingBot-World Driving Demo") as demo:
        gr.Markdown("# LingBot-World Interactive Driving Demo")
        gr.Markdown(
            "Upload a dashcam image, enter a prompt, then use the WASD "
            "buttons to generate driving video chunks with KV-cached "
            "causal inference.")

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Starting Image")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Dashcam footage of a vehicle driving on a road "
                          "during daytime. Clear weather, suburban environment.")
                upload_btn = gr.Button("Load Image + Prompt", variant="primary")

                gr.Markdown("### Controls")
                with gr.Row():
                    gr.Button("").click(lambda: None, outputs=[])  # spacer
                    btn_w = gr.Button("W (Forward)", variant="primary")
                    gr.Button("").click(lambda: None, outputs=[])
                with gr.Row():
                    btn_a = gr.Button("A (Left)")
                    btn_coast = gr.Button("Coast")
                    btn_d = gr.Button("D (Right)")
                with gr.Row():
                    gr.Button("").click(lambda: None, outputs=[])
                    btn_s = gr.Button("S (Reverse)")
                    gr.Button("").click(lambda: None, outputs=[])
                with gr.Row():
                    btn_reset = gr.Button("Reset", variant="stop")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Current View", type="pil")
                status_bar = gr.Textbox(label="Status", interactive=False)

        # Wire up events
        upload_btn.click(on_upload,
                         inputs=[img_input, prompt_input],
                         outputs=[output_image, status_bar])

        btn_w.click(on_forward, outputs=[output_image, status_bar])
        btn_a.click(on_left, outputs=[output_image, status_bar])
        btn_d.click(on_right, outputs=[output_image, status_bar])
        btn_s.click(on_reverse, outputs=[output_image, status_bar])
        btn_coast.click(on_coast, outputs=[output_image, status_bar])
        btn_reset.click(on_reset, outputs=[output_image, status_bar])

    return demo


def main():
    parser = argparse.ArgumentParser(
        description='Gradio interactive driving demo with KV-cached causal generation')
    parser.add_argument('--ckpt_dir', required=True,
                        help='Path to lingbot-world-base-cam weights')
    parser.add_argument('--size', default='480*832',
                        choices=['480*832', '720*1280'])
    parser.add_argument('--frame_num', type=int, default=17,
                        help='Frames per chunk (must be 4n+1)')
    parser.add_argument('--steps', type=int, default=6,
                        help='Denoising steps per chunk')
    parser.add_argument('--max_cache_chunks', type=int, default=4,
                        help='Max chunks to keep in KV cache')
    parser.add_argument('--max_area', type=int, default=320 * 576,
                        help='Max pixel area (320p default)')
    parser.add_argument('--speed', type=float, default=0.3)
    parser.add_argument('--turn_rate', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--share', action='store_true',
                        help='Create a public Gradio link')
    parser.add_argument('--port', type=int, default=7860)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    demo = create_demo(args)
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share)


if __name__ == '__main__':
    main()
