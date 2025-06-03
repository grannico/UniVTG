import os
import time
import torch
import gradio as gr
import numpy as np
import argparse
import subprocess
from run_on_video import clip, vid2clip, txt2clip

from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array

# ------------------ PARSE ARGS ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./tmp')
parser.add_argument('--resume', type=str, default='./results/omni/model_best.ckpt')
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# ------------------ LOAD CLIP ------------------
clip_model, _ = clip.load("ViT-B/32", device=args.gpu_id, jit=False)

# ------------------ SETUP MODEL ------------------
def load_model():
    opt = TestOptions().parse(args)
    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]
    model, _, _, _ = setup_model(opt)
    return model

vtg_model = load_model()

# ------------------ UTILS ------------------
clip_len = 2

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(save_dir):
    vid = np.load(os.path.join(save_dir, 'vid.npz'))['features'].astype(np.float32)
    txt = np.load(os.path.join(save_dir, 'txt.npz'))['features'].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    ctx_l = vid.shape[0]
    timestamp = ((torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    tef = torch.stack([tef_st, tef_ed], dim=1)
    vid = torch.cat([vid, tef], dim=1)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, save_dir, query):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()

    pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    pred_confidence = pred_logits

    top1_window = pred_windows[torch.argmax(pred_confidence)].tolist()
    mr_res = " - ".join([convert_to_hms(int(i)) for i in top1_window])
    hl_res = convert_to_hms(torch.argmax(pred_saliency) * clip_len)

    response = f"For query: {query}\nThe Top-1 interval is: {mr_res}\nThe Top-1 highlight is: {hl_res}"
    print(f"[DEBUG] Top-1 interval: {mr_res}")
    print(f"[DEBUG] Top-1 highlight: {hl_res}")
    return response

def extract_vid(vid_path, state):
    vid2clip(clip_model, vid_path, args.save_dir)
    state['messages'].append({"role": "user", "content": "Finish extracting video features."})
    state['messages'].append({"role": "system", "content": "Please enter the text query."})
    return '', state['messages'], state

def extract_txt(txt):
    print(f"[DEBUG] Estrazione testo: {txt}")
    txt2clip(clip_model, txt, args.save_dir)
    if os.path.exists(os.path.join(args.save_dir, 'txt.npz')):
        print("[DEBUG] txt.npz creato con successo.")
    else:
        print("[ERROR] txt.npz NON creato.")

def download_video(url, save_dir='./examples', size=768):
    save_path = f'{save_dir}/{url}.mp4'
    cmd = f'yt-dlp -S ext:mp4:m4a --throttled-rate 5M -f "best[width<={size}][height<={size}]" --output {save_path} --merge-output-format mp4 https://www.youtube.com/embed/{url}'
    if not os.path.exists(save_path):
        subprocess.call(cmd, shell=True)
    return save_path

def get_empty_state():
    return {"total_tokens": 0, "messages": []}

def submit_message(prompt, state):
    history = state['messages']
    print(f"[DEBUG] Prompt ricevuto: {prompt}")

    if not prompt:
        return '', history, state

    try:
        history.append({"role": "user", "content": prompt})
        extract_txt(prompt)
        response = forward(vtg_model, args.save_dir, prompt)
        history.append({"role": "system", "content": response})
        print(f"[DEBUG] Risposta generata: {response}")
    except Exception as e:
        error_msg = f"Error: {e}"
        print(f"[ERROR] {error_msg}")
        history.append({"role": "system", "content": error_msg})

    return '', history, state

def clear_conversation():
    return '', None, [], get_empty_state()

def subvid_fn(vid):
    return download_video(vid)

# ------------------ GRADIO UI ------------------
css = """
#col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
#video_inp {min-height: 100px}
#chatbox {min-height: 100px;}
#header {text-align: center;}
#hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
.message { font-size: 1.2em; }
"""

with gr.Blocks(css=css) as demo:
    state = gr.State(get_empty_state())

    with gr.Column(elem_id="col-container"):
        gr.Markdown("## 🤖️ UniVTG: Unified Video-Language Temporal Grounding", elem_id="header")
        gr.Markdown("https://github.com/showlab/UniVTG")

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                gr.Markdown("👋 **Step1**: Upload video or enter a YouTube ID below", elem_id="hint")
                with gr.Row():
                    video_id = gr.Textbox(placeholder="YouTube video ID")
                    vidsub_btn = gr.Button("(Optional) Submit Youtube id")

            with gr.Column():
                vid_ext = gr.Button("Step2: Extract video features")
                chatbot = gr.Chatbot(elem_id="chatbox", type="messages")
                input_message = gr.Textbox(show_label=False, placeholder="Enter your text query", visible=True)
                btn_submit = gr.Button("Step3: Submit text query")
                btn_clear_conversation = gr.Button("🔃 Clear")

        examples = gr.Examples(
            examples=[["./examples/charades.mp4"]],
            inputs=[video_inp],
        )

    gr.HTML('<br><br><br><center><a href="https://huggingface.co/spaces/anzorq/chatgpt-demo?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a></center>')

    btn_submit.click(submit_message, [input_message, state], [input_message, chatbot, state])
    input_message.submit(submit_message, [input_message, state], [input_message, chatbot, state])
    btn_clear_conversation.click(clear_conversation, [], [input_message, video_inp, chatbot, state])
    vid_ext.click(extract_vid, [video_inp, state], [input_message, chatbot, state])
    vidsub_btn.click(subvid_fn, [video_id], [video_inp])

demo.queue()
demo.launch(server_port=2253, debug=True, share=True)
