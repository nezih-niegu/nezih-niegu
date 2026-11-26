"""Visual abstracts for the profile README, one per research area (Plotly + Kaleido).
Each card shows a project's pipeline as a flow of steps and its headline result.
Run:  python scripts/make_abstracts.py   ->  assets/area_*.png"""
import os

import plotly.graph_objects as go

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
INK, INK2, LINE, CARD = "#1f2328", "#59636e", "#d1d9e0", "#f6f8fa"
FONT = "Arial, Helvetica, sans-serif"

AREAS = {
    "neuro": ("Neuro & medical AI", "brain and body signals → models that are tested out of sample", "#8250df", [
        ("Neurovision", ["EEG<br>ds002778", "MI-weighted<br>channel graph", "hide one<br>channel", "reconstruct<br>held-out block"],
         "Do high-weight sources beat low-weight ones out of sample?"),
        ("neuroimaging", ["raw<br>DICOM", "FreeSurfer 8<br>recon-all", "normative<br>models", "clinical<br>PDF report"],
         "Percentile-for-age vs ICBM / PPMI / ADNI controls"),
        ("heart-pignn", ["ECG<br>beat", "SA → AV →<br>His–Purkinje", "physics-informed<br>GNN", "AAMI class<br>+ waveform"],
         "Signal propagates over the cardiac conduction graph (MIT-BIH)"),
        ("Trismegisto", ["CT<br>DICOM", "Total-<br>Segmentator", "aorta<br>mask", "radiomic<br>features"],
         "Aortic morphology: volume, surface area, diameters"),
        ("stretch-posture-coach", ["D435i<br>camera", "MediaPipe<br>Pose", "RULA / REBA<br>L5/S1 load", "spoken<br>advice"],
         "Runs on a Stretch 3 robot: local LLM + offline voice, no cloud"),
    ]),
    "robotics": ("Robotics & control", "from optimal control to learned policies, in simulation and on hardware", "#0969da", [
        ("quadruped-optimal-control", ["quadruped<br>MuJoCo", "external<br>disturbances", "PMP · LQG<br>· MPC", "velocity<br>tracking"],
         "Three optimal controllers compared under the same pushes"),
        ("anymal_d_ppo", ["ANYmal D<br>MuJoCo", "12 leg<br>joints", "PPO", "locomotion<br>policy"],
         "Reinforcement-learned locomotion, tracked on Hugging Face"),
        ("psyonic_ros", ["webcam<br>hand", "landmarks<br>+ Kalman", "MuJoCo<br>safety filter", "Ability Hand<br>+ xArm Lite 6"],
         "The robot only receives poses the simulation actually reached"),
        ("auv_ftc_ws", ["AUV<br>Gazebo", "actuator<br>faults", "T-S fuzzy<br>control", "pseudo-inverse<br>QP allocation"],
         "Fault-tolerant underwater control in ROS 2 Humble"),
        ("pb-j_control", ["PuzzleBot<br>diff-drive", "terrain<br>perturbations", "5 port-<br>Hamiltonian ctrl", "Lyapunov<br>+ phase portraits"],
         "Nonholonomic control with stability analysis"),
        ("g1_cuda_lidar_benchmark", ["Unitree G1<br>head LiDAR", "CPU · naive<br>· optimised CUDA", "bounded<br>voxel map", "ROS 2<br>Humble + Jazzy"],
         "GPU LiDAR filtering benchmark on a humanoid"),
        ("lerobot-to-cosmos", ["LeRobot v3<br>dataset", "100 episodes<br>× 4 cameras", "Cosmos<br>video-SFT", "validated vs<br>reference"],
         "Robot data → NVIDIA Cosmos world-model training format"),
        ("RobocupVision-SAM", ["match<br>video", "SAM<br>tracking", "DINOv2<br>embeddings", "teams, ball<br>& field"],
         "Teams found by visual similarity, not hard-coded colours"),
        ("Segkit", ["images<br>+ masks", "UNet<br>training", "ONNX →<br>TensorRT", "Jetson Nano<br>15 fps"],
         "Semantic segmentation from training to a live camera"),
    ]),
    "hpc": ("HPC, systems & quantitative experiments", "measuring what is actually fast, and what a model actually explains", "#bc4c00", [
        ("rust-without-shortcuts", ["9 systems<br>languages", "8<br>benchmarks", "AI-kernel<br>suite", "12-language<br>superset"],
         "Rust Among Giants: languages for AI infrastructure"),
        ("parallel-fashion-intelligence", ["router<br>MLP", "3 specialists<br>+ singleton", "OpenMP + AVX-512<br>· MPI · CUDA", "same frozen<br>checkpoints"],
         "90.15% flat reference · 97.39% group routing accuracy"),
        ("mcp-mpi-benchmark", ["agent ↔<br>tool calls", "MCP-style<br>transport", "MPI<br>transport", "end-to-end +<br>microbenchmark"],
         "Is MPI a useful middleware for agent/tool communication?"),
        ("GPU-Compiler-Optimization", ["tensor<br>program", "algorithmic<br>analysis", "automatic<br>optimisation", "optimised<br>kernels"],
         "A compiler for tensor operations"),
        ("broker-realdata-test", ["Binance<br>BTC/USDT ticks", "closed-form<br>Nash broker", "learned<br>brokers", "144 held-out<br>hours"],
         "Nash broker wins only at 1 bps; assumptions rejected by data"),
        ("qse3-sim", ["Min(1,3)<br>× SU(N)", "neutrino-like<br>oscillation + MSW", "11 entangled<br>robots", "Lorenz<br>guidance"],
         "177,147-amplitude joint state over 210,000 km"),
    ]),
}


def card(fig, x0, y0, w, h, color, name, steps, result):
    fig.add_shape(type="rect", x0=x0, x1=x0 + w, y0=y0, y1=y0 + h, line=dict(color=LINE, width=1.2), fillcolor=CARD)
    fig.add_shape(type="rect", x0=x0, x1=x0 + 6, y0=y0, y1=y0 + h, line_width=0, fillcolor=color)
    fig.add_annotation(x=x0 + 20, y=y0 + h - 16, text=f"<b>{name}</b>", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(size=17, color=INK))
    n, pad, gap = len(steps), 20, 18
    cw = (w - 2 * pad - (n - 1) * gap) / n
    cy0, cy1 = y0 + 50, y0 + h - 52
    for i, s in enumerate(steps):
        cx = x0 + pad + i * (cw + gap)
        last = i == n - 1
        fig.add_shape(type="rect", x0=cx, x1=cx + cw, y0=cy0, y1=cy1, line=dict(color=color, width=1.5),
                      fillcolor=color if last else "white", opacity=1)
        fig.add_annotation(x=cx + cw / 2, y=(cy0 + cy1) / 2, text=s, showarrow=False, align="center",
                           font=dict(size=12, color="white" if last else INK))
        if i < n - 1:
            fig.add_annotation(x=cx + cw + gap - 2, y=(cy0 + cy1) / 2, ax=cx + cw + 2, ay=(cy0 + cy1) / 2,
                               xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=2,
                               arrowsize=1, arrowwidth=1.8, arrowcolor=color)
    fig.add_annotation(x=x0 + 20, y=y0 + 16, text=result, showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12.5, color=INK2))


def area(key):
    title, sub, color, projects = AREAS[key]
    cols, W, cw, ch, gx, gy, top, side = 3, 1755, 570, 190, 15, 15, 92, 7.5
    rows = -(-len(projects) // cols)
    H = top + rows * (ch + gy) + 6
    fig = go.Figure()
    fig.update_layout(width=W, height=H, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white",
                      plot_bgcolor="white", showlegend=False, font=dict(family=FONT),
                      xaxis=dict(range=[0, W], visible=False, fixedrange=True),
                      yaxis=dict(range=[0, H], visible=False, fixedrange=True))
    fig.add_shape(type="rect", x0=side, x1=side + 6, y0=H - 72, y1=H - 18, line_width=0, fillcolor=color)
    fig.add_annotation(x=side + 20, y=H - 14, text=f"<b>{title}</b>", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(size=26, color=INK))
    fig.add_annotation(x=side + 20, y=H - 52, text=sub, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=15, color=INK2))
    for i, (name, steps, result) in enumerate(projects):
        r, c = divmod(i, cols)
        x0 = side + c * (cw + gx)
        y0 = H - top - (r + 1) * ch - r * gy
        card(fig, x0, y0, cw, ch, color, name, steps, result)
    os.makedirs(OUT, exist_ok=True)
    fig.write_image(os.path.join(OUT, f"area_{key}.png"), scale=2)


if __name__ == "__main__":
    for k in AREAS:
        area(k)
    print("ok")
