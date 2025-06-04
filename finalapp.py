# gradapp_with_history.py
# 3D-DiT Generator with Editable STL Viewer, Download, and Version History

import gradio as gr
import base64, html, tempfile
from PIL import Image
import trimesh
import edit  # your editing module
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FaceReducer, FloaterRemover, DegenerateFaceRemover
)

# 1️⃣ Load once
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2"
)
rembg = BackgroundRemover()

# Helper to build viewer HTML and export STL file

def build_viewer_and_file(mesh):
    # export to a persistent temp .stl
    stl_temp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh.export(stl_temp.name)
    stl_path = stl_temp.name
    stl_temp.close()

    # encode for inline viewer
    with open(stl_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    viewer_html = f"""<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"UTF-8\"><title>3D Viewer</title>
<script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js\"></script>
<script src=\"https://cdn.jsdelivr.net/npm/three@0.134/examples/js/loaders/STLLoader.js\"></script>
<script src=\"https://cdn.jsdelivr.net/npm/three@0.134/examples/js/controls/OrbitControls.js\"></script>
<style>body{{margin:0;overflow:hidden}}#viewer{{width:100vw;height:100vh}}</style>
</head><body><div id=\"viewer\"></div><script>
(function(){{
  const bin=atob("{b64}"), L=bin.length, buf=new Uint8Array(L);
  for(let i=0;i<L;i++) buf[i]=bin.charCodeAt(i);
  const arrayBuffer=buf.buffer;
  const container=document.getElementById('viewer'), w=container.clientWidth, h=container.clientHeight;
  const scene=new THREE.Scene(); scene.add(new THREE.AmbientLight(0xffffff,0.8));
  [[5,5,5],[-5,5,5],[5,-5,5],[5,5,-5]].forEach(p=>{{ const dl=new THREE.DirectionalLight(0xffffff,0.5); dl.position.set(...p); scene.add(dl); }});
  const camera=new THREE.PerspectiveCamera(75,w/h,0.1,1000), renderer=new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(w,h); container.appendChild(renderer.domElement);
  const loader=new THREE.STLLoader(), geom=loader.parse(arrayBuffer);
  geom.computeBoundingBox();
  const c=geom.boundingBox.getCenter(new THREE.Vector3()); geom.translate(-c.x,-c.y,-c.z);
  const mesh=new THREE.Mesh(geom, new THREE.MeshStandardMaterial({{color:0x606060}})); scene.add(mesh);
  const size=geom.boundingBox.getSize(new THREE.Vector3()).length(), fov=camera.fov*Math.PI/180;
  camera.position.z=Math.abs(size/Math.sin(fov/2))/2;
  const controls=new THREE.OrbitControls(camera,renderer.domElement);
  (function animate(){{ requestAnimationFrame(animate); mesh.rotation.y+=0.005; controls.update(); renderer.render(scene,camera); }})();
}})();
</script></body></html>"""
    iframe = f'<iframe srcdoc="{html.escape(viewer_html)}" width="100%" height="400" style="border:none;"></iframe>'
    return iframe, stl_path

# 2️⃣ Handlers

def generate_and_view(image_path):
    # initial generation and reset history
    img = Image.open(image_path).convert("RGB")
    img = rembg(img)
    mesh = pipeline(image=img)[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    iframe, stl_path = build_viewer_and_file(mesh)
    # initialize history: only the original
    state = {"history": [stl_path], "index": 0}
    return iframe, stl_path, state, "✅ Model created"


def handle_edit(cmd, state):
    hist = state["history"]
    idx = state["index"]
    # load current mesh
    mesh = trimesh.load_mesh(hist[idx])
    mesh, success = claude_edit.process_command(cmd, mesh)
    if not success:
        return gr.update(), None, state, "⚠️ Edit failed"

    iframe, new_stl = build_viewer_and_file(mesh)
    # discard any redo history and append
    new_hist = hist[: idx + 1] + [new_stl]
    new_idx = len(new_hist) - 1
    new_state = {"history": new_hist, "index": new_idx}
    return iframe, new_stl, new_state, f"✅ Applied: {cmd.strip()}"


def handle_undo(state):
    hist = state["history"]
    idx = state["index"]
    if idx <= 0:
        return gr.update(), None, state, "⚠️ No earlier version"
    new_idx = idx - 1
    iframe, stl_path = build_viewer_and_file(trimesh.load_mesh(hist[new_idx]))
    new_state = {"history": hist, "index": new_idx}
    return iframe, stl_path, new_state, f"⏪ Reverted to version {new_idx}"


def handle_redo(state):
    hist = state["history"]
    idx = state["index"]
    if idx >= len(hist) - 1:
        return gr.update(), None, state, "⚠️ No later version"
    new_idx = idx + 1
    iframe, stl_path = build_viewer_and_file(trimesh.load_mesh(hist[new_idx]))
    new_state = {"history": hist, "index": new_idx}
    return iframe, stl_path, new_state, f"⏩ Advanced to version {new_idx}"

# 3️⃣ UI
with gr.Blocks() as demo:
    gr.Markdown("## Image to 3D Model Generator With Natural Language Editing")
    with gr.Row():
        with gr.Column():
            inp     = gr.Image(type="filepath", label="Upload Object Image")
            gen_b   = gr.Button("Create 3D Model")
        with gr.Column():
            viewer  = gr.HTML("<div style='color:#999;'>No model yet…</div>")
            download = gr.File(label="Download STL")
            state    = gr.State()
            feedback = gr.Textbox(label="Feedback", interactive=False)
            with gr.Row():
                undo_b = gr.Button("⏪ Undo")
                redo_b = gr.Button("⏩ Redo")
            cmd_in   = gr.Textbox(placeholder="extend width by 5cm…", lines=2)
            edit_b   = gr.Button("Apply Edit")

    gen_b.click(
        fn=generate_and_view,
        inputs=inp,
        outputs=[viewer, download, state, feedback]
    )
    edit_b.click(
        fn=handle_edit,
        inputs=[cmd_in, state],
        outputs=[viewer, download, state, feedback]
    )
    undo_b.click(
        fn=handle_undo,
        inputs=state,
        outputs=[viewer, download, state, feedback]
    )
    redo_b.click(
        fn=handle_redo,
        inputs=state,
        outputs=[viewer, download, state, feedback]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
