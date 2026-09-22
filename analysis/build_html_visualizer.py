import json
from pathlib import Path

def build_visualizer():
    with open("neural_network_data.json", "r") as f:
        network_data_str = f.read()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Unitree G1 — Neural Network Policy Activation Visualizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #07090e;
      --bg-panel: rgba(14, 20, 32, 0.75);
      --border-panel: rgba(0, 242, 254, 0.18);
      --accent-cyan: #00f2fe;
      --accent-emerald: #10b981;
      --accent-violet: #a855f7;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.07) 0%, transparent 45%),
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
    }}

    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 18px 32px;
      background: var(--bg-panel);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-panel);
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .badge-live {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}

    .badge-live::before {{
      content: '';
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 1.8s infinite;
    }}

    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(1.4); opacity: 0.6; }}
    }}

    h1 {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, var(--accent-cyan) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .subtitle {{
      font-size: 12px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }}

    .metrics-bar {{
      display: flex;
      gap: 24px;
      align-items: center;
    }}

    .metric-card {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.06);
      padding: 6px 14px;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }}

    .metric-title {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
    }}

    .metric-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 16px;
      font-weight: 600;
      color: var(--accent-cyan);
    }}

    main {{
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px 24px;
      gap: 16px;
    }}

    .canvas-container {{
      flex: 1;
      position: relative;
      background: rgba(10, 14, 22, 0.85);
      border: 1px solid var(--border-panel);
      border-radius: 12px;
      overflow: hidden;
      min-height: 560px;
      box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.6);
    }}

    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}

    .layer-legend {{
      position: absolute;
      top: 14px;
      left: 20px;
      right: 20px;
      display: flex;
      justify-content: space-between;
      pointer-events: none;
    }}

    .layer-tag {{
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-main);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }}

    .control-panel {{
      background: var(--bg-panel);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border-panel);
      border-radius: 12px;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      gap: 20px;
    }}

    .btn {{
      background: linear-gradient(135deg, rgba(0, 242, 254, 0.2), rgba(0, 120, 255, 0.3));
      border: 1px solid rgba(0, 242, 254, 0.4);
      color: #ffffff;
      padding: 9px 20px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }}

    .btn:hover {{
      background: linear-gradient(135deg, rgba(0, 242, 254, 0.35), rgba(0, 120, 255, 0.45));
      border-color: var(--accent-cyan);
      box-shadow: 0 0 16px rgba(0, 242, 254, 0.3);
      transform: translateY(-1px);
    }}

    .timeline-slider {{
      flex: 1;
      accent-color: var(--accent-cyan);
      cursor: pointer;
      height: 6px;
      border-radius: 3px;
    }}

    .time-display {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: var(--text-muted);
      min-width: 130px;
    }}

    .color-legend {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 11px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }}

    .gradient-bar {{
      width: 100px;
      height: 8px;
      border-radius: 4px;
      background: linear-gradient(to right, #1e293b 0%, #0369a1 30%, #00f2fe 70%, #f59e0b 100%);
    }}

    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(11, 17, 29, 0.95);
      border: 1px solid var(--accent-cyan);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      color: #ffffff;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6), 0 0 15px rgba(0, 242, 254, 0.2);
      display: none;
      z-index: 100;
      min-width: 180px;
    }}

    .tooltip-title {{
      font-weight: 700;
      color: var(--accent-cyan);
      margin-bottom: 4px;
      font-family: 'JetBrains Mono', monospace;
    }}

    .tooltip-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: #34d399;
    }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="badge-live">Live Synapse Feed</div>
      <div>
        <h1>Unitree G1 Humanoid — Neural Network Policy Visualizer</h1>
        <div class="subtitle">Architecture: 49 (Input) → 256 (ELU) → 256 (ELU) → 128 (ELU) → 12 (Joint Positions)</div>
      </div>
    </div>

    <div class="metrics-bar">
      <div class="metric-card">
        <span class="metric-title">Command Speed</span>
        <span class="metric-val" id="disp-cmd">0.40 m/s</span>
      </div>
      <div class="metric-card">
        <span class="metric-title">Actual Velocity</span>
        <span class="metric-val" id="disp-vel">0.41 m/s</span>
      </div>
      <div class="metric-card">
        <span class="metric-title">Distance Walked</span>
        <span class="metric-val" id="disp-dist">0.00 m</span>
      </div>
      <div class="metric-card">
        <span class="metric-title">Gait Phase</span>
        <span class="metric-val" id="disp-phase">0.00</span>
      </div>
    </div>
  </header>

  <main>
    <div class="canvas-container">
      <canvas id="network-canvas"></canvas>
      <div class="layer-legend">
        <span class="layer-tag">Input Layer (49)</span>
        <span class="layer-tag">Hidden Layer 1 (256 ELU)</span>
        <span class="layer-tag">Hidden Layer 2 (256 ELU)</span>
        <span class="layer-tag">Hidden Layer 3 (128 ELU)</span>
        <span class="layer-tag">Output Actions (12 Joints)</span>
      </div>
      <div class="tooltip" id="tooltip">
        <div class="tooltip-title" id="tt-title">Neuron #</div>
        <div>Signal: <span id="tt-name">-</span></div>
        <div>Activation: <span class="tooltip-val" id="tt-val">0.00</span></div>
      </div>
    </div>

    <div class="control-panel">
      <button class="btn" id="btn-play">
        <span id="play-icon">⏸</span> <span id="play-text">Pause</span>
      </button>

      <input type="range" class="timeline-slider" id="time-slider" min="0" max="249" value="0">

      <div class="time-display" id="time-display">
        t = 0.00s / 5.00s
      </div>

      <div class="color-legend">
        <span>-1.0 (Inhibited)</span>
        <div class="gradient-bar"></div>
        <span>+15.0 (Excited)</span>
      </div>
    </div>
  </main>

  <script>
    const NET_DATA = {network_data_str};

    const canvas = document.getElementById('network-canvas');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('time-slider');
    const btnPlay = document.getElementById('btn-play');
    const playIcon = document.getElementById('play-icon');
    const playText = document.getElementById('play-text');
    const timeDisplay = document.getElementById('time-display');
    const tooltip = document.getElementById('tooltip');

    const dispCmd = document.getElementById('disp-cmd');
    const dispVel = document.getElementById('disp-vel');
    const dispDist = document.getElementById('disp-dist');
    const dispPhase = document.getElementById('disp-phase');

    let currentFrameIdx = 0;
    let isPlaying = true;
    let animationTimer = null;
    let hoveredNode = null;

    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }}

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Map activation to glowing color
    function getActivationColor(val, isOutput = false) {{
      if (isOutput) {{
        // Output is between -1.0 and 1.0
        if (val > 0) {{
          const a = Math.min(1.0, val * 1.5);
          return `rgba(16, 185, 129, ${{0.3 + a * 0.7}})`; // Green for forward action
        }} else {{
          const a = Math.min(1.0, Math.abs(val) * 1.5);
          return `rgba(244, 63, 94, ${{0.3 + a * 0.7}})`; // Rose for backward action
        }}
      }}

      // ELU activation: >= -1.0
      if (val < 0) {{
        // Inhibited / negative
        const intensity = Math.abs(val); // 0 to 1
        return `rgba(30, 41, 59, ${{0.4 + intensity * 0.4}})`;
      }} else {{
        // Positive excitation
        if (val < 2.0) {{
          const t = val / 2.0;
          return `rgba(3, 169, 244, ${{0.5 + t * 0.3}})`; // Cyan
        }} else if (val < 8.0) {{
          const t = (val - 2.0) / 6.0;
          return `rgba(0, 242, 254, ${{0.8 + t * 0.2}})`; // Bright cyan/white
        }} else {{
          return `rgba(245, 158, 11, 1.0)`; // Gold overload
        }}
      }}
    }}

    // Compute node coordinates
    function getNodePositions(w, h) {{
      const layers = [49, 256, 256, 128, 12];
      const layerX = [
        w * 0.08,
        w * 0.30,
        w * 0.52,
        w * 0.74,
        w * 0.92
      ];

      const nodeCoords = [];

      // Layer 0: 49 nodes in single column
      const l0 = [];
      const top0 = 60, bot0 = h - 30;
      for (let i = 0; i < 49; i++) {{
        const y = top0 + (i / 48) * (bot0 - top0);
        l0.push({{ x: layerX[0], y, r: 4.5, layer: 0, idx: i }});
      }}
      nodeCoords.push(l0);

      // Layer 1: 256 nodes in 4 columns of 64
      const l1 = [];
      const cols1 = 4, perCol1 = 64;
      for (let i = 0; i < 256; i++) {{
        const c = Math.floor(i / perCol1);
        const r = i % perCol1;
        const x = layerX[1] + (c - 1.5) * 18;
        const y = top0 + (r / (perCol1 - 1)) * (bot0 - top0);
        l1.push({{ x, y, r: 3.2, layer: 1, idx: i }});
      }}
      nodeCoords.push(l1);

      // Layer 2: 256 nodes in 4 columns of 64
      const l2 = [];
      for (let i = 0; i < 256; i++) {{
        const c = Math.floor(i / perCol1);
        const r = i % perCol1;
        const x = layerX[2] + (c - 1.5) * 18;
        const y = top0 + (r / (perCol1 - 1)) * (bot0 - top0);
        l2.push({{ x, y, r: 3.2, layer: 2, idx: i }});
      }}
      nodeCoords.push(l2);

      // Layer 3: 128 nodes in 2 columns of 64
      const l3 = [];
      const cols3 = 2, perCol3 = 64;
      for (let i = 0; i < 128; i++) {{
        const c = Math.floor(i / perCol3);
        const r = i % perCol3;
        const x = layerX[3] + (c - 0.5) * 22;
        const y = top0 + (r / (perCol3 - 1)) * (bot0 - top0);
        l3.push({{ x, y, r: 3.8, layer: 3, idx: i }});
      }}
      nodeCoords.push(l3);

      // Layer 4: 12 output nodes
      const l4 = [];
      const top4 = 90, bot4 = h - 60;
      for (let i = 0; i < 12; i++) {{
        const y = top4 + (i / 11) * (bot4 - top4);
        l4.push({{ x: layerX[4], y, r: 6.5, layer: 4, idx: i }});
      }}
      nodeCoords.push(l4);

      return nodeCoords;
    }}

    function drawNetwork() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      const frame = NET_DATA.frames[currentFrameIdx];
      const coords = getNodePositions(w, h);

      // 1. Draw Synapses
      const syn = NET_DATA.architecture.synapses;
      function drawSynapseGroup(sList, lFrom, lTo) {{
        for (let k = 0; k < sList.length; k++) {{
          const [fromIdx, toIdx, weight] = sList[k];
          const p1 = coords[lFrom][fromIdx];
          const p2 = coords[lTo][toIdx];
          if (!p1 || !p2) continue;

          const act1 = frame[`L${{lFrom}}`][fromIdx];
          const alpha = Math.min(0.35, Math.abs(weight) * 0.12 * (act1 > 0 ? 1.5 : 0.6));

          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.bezierCurveTo(p1.x + (p2.x - p1.x)*0.5, p1.y, p1.x + (p2.x - p1.x)*0.5, p2.y, p2.x, p2.y);
          ctx.strokeStyle = weight > 0 ? `rgba(0, 242, 254, ${{alpha}})` : `rgba(168, 85, 247, ${{alpha}})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }}
      }}

      drawSynapseGroup(syn.s01, 0, 1);
      drawSynapseGroup(syn.s12, 1, 2);
      drawSynapseGroup(syn.s23, 2, 3);
      drawSynapseGroup(syn.s34, 3, 4);

      // 2. Draw Nodes
      for (let l = 0; l < coords.length; l++) {{
        const actList = frame[`L${{l}}`];
        const isOut = (l === 4);

        for (let i = 0; i < coords[l].length; i++) {{
          const node = coords[l][i];
          const val = actList[i];
          const color = getActivationColor(val, isOut);

          // Glow bloom if active
          if (val > 1.5 || (isOut && Math.abs(val) > 0.4)) {{
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.r * 2.4, 0, Math.PI * 2);
            ctx.fillStyle = color.replace('rgba', 'rgba').replace(/[\d\.]+\)$/, '0.15)');
            ctx.fill();
          }}

          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.lineWidth = 0.6;
          ctx.stroke();

          // Text labels for output layer
          if (l === 4) {{
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillStyle = '#cbd5e1';
            ctx.textAlign = 'left';
            const name = NET_DATA.architecture.output_names[i];
            ctx.fillText(`${{name}} (${{val > 0 ? '+' : ''}}${{val.toFixed(2)}})`, node.x + 14, node.y + 3);
          }}
        }}
      }}

      // 3. Highlight hovered node
      if (hoveredNode) {{
        ctx.beginPath();
        ctx.arc(hoveredNode.x, hoveredNode.y, hoveredNode.r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }}

      // Update HUD metrics
      dispCmd.textContent = `${{frame.cmd_vx.toFixed(2)}} m/s`;
      dispVel.textContent = `${{frame.actual_vx.toFixed(2)}} m/s`;
      dispDist.textContent = `${{frame.distance.toFixed(2)}} m`;
      dispPhase.textContent = `${{frame.gait_phase.toFixed(2)}}`;
      timeDisplay.textContent = `t = ${{frame.t.toFixed(2)}}s / 5.00s (Step ${{frame.step}})`;
      slider.value = currentFrameIdx;
    }}

    // Animation Loop
    function tick() {{
      if (isPlaying) {{
        currentFrameIdx = (currentFrameIdx + 1) % NET_DATA.frames.length;
        drawNetwork();
      }}
    }}

    animationTimer = setInterval(tick, 40); // 25 FPS playback

    btnPlay.addEventListener('click', () => {{
      isPlaying = !isPlaying;
      playIcon.textContent = isPlaying ? '⏸' : '▶';
      playText.textContent = isPlaying ? 'Pause' : 'Play';
    }});

    slider.addEventListener('input', (e) => {{
      currentFrameIdx = parseInt(e.target.value);
      drawNetwork();
    }});

    // Hover tooltip
    canvas.addEventListener('mousemove', (e) => {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const coords = getNodePositions(rect.width, rect.height);
      hoveredNode = null;

      for (let l = 0; l < coords.length; l++) {{
        for (let i = 0; i < coords[l].length; i++) {{
          const n = coords[l][i];
          const dist = Math.hypot(mx - n.x, my - n.y);
          if (dist < n.r + 5) {{
            hoveredNode = n;
            break;
          }}
        }}
        if (hoveredNode) break;
      }}

      if (hoveredNode) {{
        const l = hoveredNode.layer;
        const i = hoveredNode.idx;
        const val = NET_DATA.frames[currentFrameIdx][`L${{l}}`][i];
        
        let signalName = `Neuron L${{l}}_${{i}}`;
        if (l === 0) signalName = NET_DATA.architecture.input_names[i] || `Input ${{i}}`;
        if (l === 4) signalName = NET_DATA.architecture.output_names[i] || `Output ${{i}}`;

        tooltip.style.display = 'block';
        tooltip.style.left = `${{e.clientX - rect.left + 15}}px`;
        tooltip.style.top = `${{e.clientY - rect.top - 20}}px`;
        document.getElementById('tt-title').textContent = `Layer ${{l}} — Node #${{i}}`;
        document.getElementById('tt-name').textContent = signalName;
        document.getElementById('tt-val').textContent = val.toFixed(3);
      }} else {{
        tooltip.style.display = 'none';
      }}

      drawNetwork();
    }});

    canvas.addEventListener('mouseleave', () => {{
      hoveredNode = null;
      tooltip.style.display = 'none';
      drawNetwork();
    }});

    drawNetwork();
  </script>
</body>
</html>
"""

    out_path = Path("neural_network_visualizer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎉 Created interactive visualizer: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    build_visualizer()
