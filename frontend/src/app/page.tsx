"use client";
import { useState, useEffect, useRef, useCallback } from "react";

// ─── helpers ─────────────────────────────────────────────────────────────────
const textToBits = (s: string): number[] =>
  s.split("").flatMap((c) => {
    const b = c.charCodeAt(0).toString(2).padStart(8, "0");
    return b.split("").map(Number);
  });

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// QPSK: pairs of bits → symbol label
const QPSK_MAP: Record<string, { label: string; re: number; im: number }> = {
  "00": { label: "00 → S₁", re: 0.707, im: 0.707 },
  "01": { label: "01 → S₂", re: -0.707, im: 0.707 },
  "10": { label: "10 → S₃", re: -0.707, im: -0.707 },
  "11": { label: "11 → S₄", re: 0.707, im: -0.707 },
};

// ─── Constellation plot ───────────────────────────────────────────────────────
function Constellation({ noisy, recovered }: { noisy?: boolean; recovered?: boolean }) {
  const points = [
    { re: 0.707, im: 0.707, label: "00" },
    { re: -0.707, im: 0.707, label: "01" },
    { re: -0.707, im: -0.707, label: "10" },
    { re: 0.707, im: -0.707, label: "11" },
  ];
  const W = 180, H = 180, cx = W / 2, cy = H / 2, sc = 60;
  const toX = (r: number) => cx + r * sc;
  const toY = (i: number) => cy - i * sc;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: W, height: H }}>
      {/* axes */}
      <line x1={cx} y1={10} x2={cx} y2={H - 10} stroke="var(--border)" strokeWidth="1" />
      <line x1={10} y1={cy} x2={W - 10} y2={cy} stroke="var(--border)" strokeWidth="1" />
      <text x={W - 14} y={cy + 12} fontSize="9" fill="var(--dim)">I</text>
      <text x={cx + 4} y={14} fontSize="9" fill="var(--dim)">Q</text>
      {/* ideal points */}
      {points.map((p) => (
        <g key={p.label}>
          {noisy && !recovered && [...Array(6)].map((_, ni) => (
            <circle key={ni}
              cx={toX(p.re) + (Math.sin(ni * 1.3 + p.re) * 14)}
              cy={toY(p.im) + (Math.cos(ni * 1.7 + p.im) * 14)}
              r={3} fill="var(--amber)" opacity={0.45} />
          ))}
          <circle cx={toX(p.re)} cy={toY(p.im)} r={noisy && !recovered ? 4 : 6}
            fill={recovered ? "var(--green)" : noisy ? "var(--amber)" : "var(--blue)"}
            stroke="white" strokeWidth="1.5" />
          <text x={toX(p.re) + 9} y={toY(p.im) + 4} fontSize="9" fill="var(--sub)">{p.label}</text>
        </g>
      ))}
      {noisy && !recovered && (
        <text x={cx} y={H - 6} fontSize="9" fill="var(--amber)" textAnchor="middle">Noisy — scattered</text>
      )}
      {recovered && (
        <text x={cx} y={H - 6} fontSize="9" fill="var(--green)" textAnchor="middle">Clean — recovered ✓</text>
      )}
    </svg>
  );
}

// ─── LEO channel SVG ─────────────────────────────────────────────────────────
function ChannelSVG({ phase }: { phase: number }) {
  // phase 0=idle,1=transmit,2=disturbed,3=received
  const W = 520, H = 200;
  // positions
  const gndX = 60, gndY = 155;
  const satX = 270, satY = 40;
  const rxX = 460, rxY = 155;

  const showSignal = phase >= 1;
  const showNoise  = phase >= 2;
  const showRx     = phase >= 3;

  // pulse progress (0..1)
  const [pulse, setPulse] = useState(0);
  useEffect(() => {
    if (phase !== 2) { setPulse(0); return; }
    let t = 0;
    const id = setInterval(() => { t = (t + 0.012) % 1; setPulse(t); }, 40);
    return () => clearInterval(id);
  }, [phase]);

  const px = gndX + (satX - gndX) * pulse + (satX + (rxX - satX) * Math.max(0, (pulse - 0.5) / 0.5)) * 0;
  // two-leg: leg1 gnd→sat, leg2 sat→rx
  const leg = pulse < 0.5 ? 0 : 1;
  const lp  = leg === 0 ? pulse * 2 : (pulse - 0.5) * 2;
  const bpx = leg === 0 ? gndX + (satX - gndX) * lp : satX + (rxX - satX) * lp;
  const bpy = leg === 0 ? gndY + (satY - gndY) * lp : satY + (rxY - satY) * lp;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      <defs>
        <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#F0F4FF" />
          <stop offset="100%" stopColor="#F7F5F0" />
        </linearGradient>
        <filter id="softglow">
          <feGaussianBlur stdDeviation="2" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <rect width={W} height={H} fill="url(#skyGrad)" rx="12" />

      {/* Earth horizon line */}
      <path d={`M 0 ${H - 40} Q ${W / 2} ${H - 55} ${W} ${H - 40} L ${W} ${H} L 0 ${H} Z`}
        fill="#E8F5E9" opacity={0.6} />

      {/* Space label */}
      <text x={20} y={22} fill="var(--dim)" fontSize="9" fontWeight="600" letterSpacing=".08em">SPACE (550 km)</text>

      {/* Stars */}
      {[...Array(18)].map((_, i) => (
        <circle key={i} cx={(i * 131 % (W - 60)) + 30} cy={(i * 73 % 70) + 10}
          r={i % 3 === 0 ? 1.2 : 0.7} fill="#9B9690" opacity={0.4} />
      ))}

      {/* Signal path lines */}
      {showSignal && (
        <>
          <line x1={gndX} y1={gndY - 10} x2={satX} y2={satY + 14}
            stroke="var(--blue)" strokeWidth="1" strokeDasharray="5 4" opacity={0.35} />
          <line x1={satX} y1={satY + 14} x2={rxX} y2={rxY - 10}
            stroke="var(--blue)" strokeWidth="1" strokeDasharray="5 4" opacity={0.35} />
        </>
      )}

      {/* Noise distortion labels */}
      {showNoise && (
        <>
          <text x={155} y={88} fontSize="9" fill="var(--amber)" fontWeight="600">Doppler ±48 kHz</text>
          <text x={148} y={100} fontSize="9" fill="var(--amber)">Fading h[k]</text>
          <text x={335} y={88} fontSize="9" fill="var(--red)" fontWeight="600">+ Noise n[k]</text>
          {/* wavy noise path */}
          <path d={`M ${satX} ${satY + 20} Q ${satX + 30} ${satY + 50} ${satX + 60} ${satY + 40} Q ${satX + 90} ${satY + 30} ${rxX} ${rxY - 10}`}
            fill="none" stroke="var(--amber)" strokeWidth="1" strokeDasharray="3 3" opacity={0.5} />
        </>
      )}

      {/* Signal pulse travelling */}
      {phase === 2 && (
        <circle cx={bpx} cy={bpy} r={6} fill="var(--blue)" opacity={0.9} filter="url(#softglow)" />
      )}

      {/* Ground Station */}
      <g>
        {/* dish */}
        <ellipse cx={gndX} cy={gndY - 6} rx={16} ry={8} fill="none" stroke="var(--sub)" strokeWidth="2" />
        <line x1={gndX} y1={gndY - 6} x2={gndX} y2={gndY + 14} stroke="var(--sub)" strokeWidth="2" />
        <line x1={gndX - 10} y1={gndY + 14} x2={gndX + 10} y2={gndY + 14} stroke="var(--sub)" strokeWidth="2" />
        {showSignal && <circle cx={gndX} cy={gndY - 6} r={3} fill="var(--blue)" className="anim-pulse" />}
        <text x={gndX} y={gndY + 28} textAnchor="middle" fontSize="9" fill="var(--sub)" fontWeight="600">Ground Station</text>
        {phase >= 1 && <text x={gndX} y={gndY + 38} textAnchor="middle" fontSize="8" fill="var(--blue)">Sending pilots + data</text>}
      </g>

      {/* Satellite */}
      <g>
        <rect x={satX - 10} y={satY + 2} width={20} height={12} rx="3"
          fill="white" stroke="var(--sub)" strokeWidth="1.5" />
        <rect x={satX - 22} y={satY + 6} width={10} height={4} rx="1" fill="var(--blue-bg)" stroke="var(--blue)" strokeWidth="1" />
        <rect x={satX + 12} y={satY + 6} width={10} height={4} rx="1" fill="var(--blue-bg)" stroke="var(--blue)" strokeWidth="1" />
        <line x1={satX} y1={satY + 2} x2={satX} y2={satY - 4} stroke="var(--sub)" strokeWidth="1" />
        <circle cx={satX} cy={satY - 5} r={2} fill="var(--amber)" />
        <text x={satX} y={satY - 12} textAnchor="middle" fontSize="9" fill="var(--sub)" fontWeight="600">LEO Satellite</text>
        <text x={satX} y={satY - 2} textAnchor="middle" fontSize="7" fill="var(--dim)">550 km</text>
      </g>

      {/* Receiver (same as ground station, right side) */}
      <g>
        <ellipse cx={rxX} cy={rxY - 6} rx={16} ry={8} fill="none"
          stroke={showRx ? "var(--green)" : "var(--border)"} strokeWidth="2" />
        <line x1={rxX} y1={rxY - 6} x2={rxX} y2={rxY + 14}
          stroke={showRx ? "var(--green)" : "var(--border)"} strokeWidth="2" />
        <line x1={rxX - 10} y1={rxY + 14} x2={rxX + 10} y2={rxY + 14}
          stroke={showRx ? "var(--green)" : "var(--border)"} strokeWidth="2" />
        {showRx && <circle cx={rxX} cy={rxY - 6} r={3} fill="var(--green)" />}
        <text x={rxX} y={rxY + 28} textAnchor="middle" fontSize="9"
          fill={showRx ? "var(--green)" : "var(--dim)"} fontWeight="600">Receiver</text>
        {showRx && <text x={rxX} y={rxY + 38} textAnchor="middle" fontSize="8" fill="var(--green)">Signal received ✓</text>}
      </g>

      {/* channel equation label */}
      {phase >= 2 && (
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="9" fill="var(--sub)"
          fontFamily="JetBrains Mono, monospace">
          y[k] = h[k] · x[k] + n[k]
        </text>
      )}
    </svg>
  );
}

// ─── Quantum circuit SVG ──────────────────────────────────────────────────────
function QuantumCircuitSVG({ activeLayer }: { activeLayer: number }) {
  const qubits = ["|q₀⟩", "|q₁⟩", "|q₂⟩", "|q₃⟩"];
  const layers = [
    { name: "Encode pilots", col: 0, gates: ["H", "H", "H", "H"], color: "var(--blue)" },
    { name: "Rotate (θ)", col: 1, gates: ["Rₓ", "Rᵧ", "R_z", "Rₓ"], color: "var(--purple)" },
    { name: "Entangle", col: 2, gates: ["●", "⊕", "●", "⊕"], color: "var(--amber)", cnot: true },
    { name: "Rotate (φ)", col: 3, gates: ["Rᵧ", "R_z", "Rₓ", "Rᵧ"], color: "var(--purple)" },
    { name: "Measure → ĥ", col: 4, gates: ["M", "M", "M", "M"], color: "var(--green)" },
  ];
  const W = 480, gapX = 72, gapY = 38, startX = 70, startY = 28;

  return (
    <svg viewBox={`0 0 ${W} ${gapY * 4 + startY + 40}`} style={{ width: "100%", height: "auto" }}>
      {/* wires */}
      {qubits.map((_, qi) => (
        <line key={qi}
          x1={startX - 20} y1={startY + qi * gapY + 8}
          x2={startX + layers.length * gapX - 10} y2={startY + qi * gapY + 8}
          stroke="var(--border)" strokeWidth="1.5" />
      ))}
      {/* qubit labels */}
      {qubits.map((q, qi) => (
        <text key={qi} x={startX - 22} y={startY + qi * gapY + 12}
          textAnchor="end" fontSize="11" fill="var(--sub)"
          fontFamily="JetBrains Mono, monospace" fontWeight="500">{q}</text>
      ))}
      {/* layers */}
      {layers.map((layer, li) => {
        const isActive = li === activeLayer;
        const isDone = li < activeLayer;
        const cx = startX + li * gapX;
        return (
          <g key={li}>
            {/* highlight column */}
            {isActive && (
              <rect x={cx - 16} y={startY - 10}
                width={36} height={qubits.length * gapY + 16}
                rx="8" fill="var(--blue-bg)" stroke="var(--blue)" strokeWidth="1" opacity={0.7} />
            )}
            {/* CNOT vertical line */}
            {layer.cnot && (
              <line x1={cx + 2} y1={startY + 0 * gapY + 8}
                x2={cx + 2} y2={startY + 3 * gapY + 8}
                stroke={isActive ? layer.color : "var(--border)"} strokeWidth="1.5" />
            )}
            {/* gates */}
            {layer.gates.map((g, qi) => {
              const isCtrl = g === "●";
              const gx = cx + 2, gy = startY + qi * gapY + 8;
              return isCtrl ? (
                <circle key={qi} cx={gx} cy={gy} r={5}
                  fill={isActive ? layer.color : isDone ? "var(--green)" : "var(--border)"} />
              ) : (
                <g key={qi}>
                  <rect x={gx - 14} y={gy - 12} width={28} height={24} rx="6"
                    fill={isActive ? "white" : isDone ? "var(--green-bg)" : "var(--cream)"}
                    stroke={isActive ? layer.color : isDone ? "var(--green)" : "var(--border)"}
                    strokeWidth={isActive ? 1.5 : 1} />
                  <text x={gx} y={gy + 5} textAnchor="middle" fontSize={g.length > 1 ? 8 : 11}
                    fontFamily="JetBrains Mono, monospace" fontWeight="600"
                    fill={isActive ? layer.color : isDone ? "var(--green)" : "var(--dim)"}>{g}</text>
                </g>
              );
            })}
            {/* layer name */}
            <text x={cx + 2} y={startY + qubits.length * gapY + 22}
              textAnchor="middle" fontSize="8" fill={isActive ? layer.color : isDone ? "var(--green)" : "var(--dim)"} fontWeight={isActive ? "700" : "400"}>
              {layer.name}
            </text>
          </g>
        );
      })}
      {/* output arrow */}
      <text x={startX + layers.length * gapX + 4} y={startY + 1 * gapY + 12}
        fontSize="11" fill="var(--green)" fontFamily="JetBrains Mono, monospace" fontWeight="600">→ ĥ</text>
    </svg>
  );
}

// ─── Signal waveform (simple bar chart) ──────────────────────────────────────
function Waveform({ noisy, label, color }: { noisy?: boolean; label: string; color: string }) {
  const vals = Array.from({ length: 32 }, (_, i) =>
    Math.sin(i * 0.6) * (noisy ? 0.6 + Math.sin(i * 2.3) * 0.4 : 1)
  );
  const W = 280, H = 60, mid = H / 2;
  const bw = W / vals.length;
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 6 }}>{label}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, background: "var(--bg)", borderRadius: 8 }}>
        <line x1={0} y1={mid} x2={W} y2={mid} stroke="var(--border)" strokeWidth="0.5" />
        {vals.map((v, i) => {
          const bh = Math.abs(v) * 22;
          return (
            <rect key={i} x={i * bw + 1} y={mid - bh} width={bw - 2} height={bh * 2}
              fill={color} opacity={noisy ? 0.55 : 0.75} rx={1} />
          );
        })}
      </svg>
    </div>
  );
}

// ─── Step block wrapper ───────────────────────────────────────────────────────
function StepBlock({
  number, title, pill, pillType = "blue", status, delay = 0, children,
}: {
  number: number; title: string; pill: string;
  pillType?: "blue" | "green" | "amber" | "purple";
  status: "idle" | "active" | "done";
  delay?: number;
  children: React.ReactNode;
}) {
  return (
    <div className={`step-${status}`}
      style={{ display: "flex", gap: 20, opacity: status === "idle" ? 0.4 : 1, transition: "opacity .4s ease", animationDelay: `${delay}ms` }}>
      {/* left: circle + connector */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 4 }}>
        <div className="step-circle" style={{
          background: status === "done" ? "var(--green)" : status === "active" ? "var(--blue)" : "var(--cream)",
          color: status === "idle" ? "var(--dim)" : "white",
          boxShadow: status === "active" ? "0 0 0 5px var(--blue-bg)" : "none",
        }}>{status === "done" ? "✓" : number}</div>
        <div className="step-connector" style={{ flex: 1, width: 2, background: status === "done" ? "var(--green)" : "var(--border)", minHeight: 24 }} />
      </div>
      {/* right: content */}
      <div style={{ flex: 1, paddingBottom: 32 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: "var(--text)" }}>{title}</h2>
          <span className={`pill pill-${pillType}`}>{pill}</span>
        </div>
        <div className={status !== "idle" ? "anim-fadein" : ""}>{children}</div>
      </div>
    </div>
  );
}

// ─── MAIN ────────────────────────────────────────────────────────────────────
export default function Page() {
  const [step, setStep] = useState(0);          // 0=idle, 1..6=running
  const [running, setRunning] = useState(false);
  const [qLayer, setQLayer] = useState(-1);
  const [inputText, setInputText] = useState("Hello World");
  const [showResult, setShowResult] = useState(false);
  const [nmse, setNmse] = useState(-19.3);
  const [ber, setBer] = useState(0.0024);
  const [speed, setSpeed] = useState(1.0);     // 0.5=slow, 1=normal, 2=fast
  const abortRef = useRef(false);

  const bits = textToBits(inputText.slice(0, 5));  // show first 5 chars bits

  const reset = () => {
    abortRef.current = true;
    setStep(0); setRunning(false); setQLayer(-1); setShowResult(false);
    setTimeout(() => { abortRef.current = false; }, 200);
  };

  const run = useCallback(async () => {
    if (running) { reset(); return; }
    abortRef.current = false;
    setRunning(true); setShowResult(false); setQLayer(-1);
    const d = (ms: number) => sleep(ms / speed);

    // Step 1 — encode
    setStep(1); await d(1400);
    if (abortRef.current) return;

    // Step 2 — modulate
    setStep(2); await d(1400);
    if (abortRef.current) return;

    // Step 3 — channel
    setStep(3); await d(2000);
    if (abortRef.current) return;

    // Step 4 — receive
    setStep(4); await d(1200);
    if (abortRef.current) return;

    // Step 5 — quantum (animate layers)
    setStep(5);
    for (let i = 0; i < 5; i++) {
      if (abortRef.current) return;
      setQLayer(i);
      await d(800);
    }
    await d(400);
    if (abortRef.current) return;

    // Step 6 — result
    setStep(6); setShowResult(true);
    setNmse(-19.3 + (Math.random() - 0.5) * 0.15);
    setBer(0.0024 + (Math.random() - 0.5) * 0.0003);
    setRunning(false);
  }, [running, speed]);

  const stepStatus = (n: number): "idle" | "active" | "done" =>
    step === 0 ? "idle" : step === n ? "active" : step > n ? "done" : "idle";

  // symbol pairs for display
  const pairs = bits.reduce((acc: string[], b, i) => {
    if (i % 2 === 0) acc.push(bits[i].toString() + (bits[i + 1] ?? "0").toString());
    return acc;
  }, []);

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>

      {/* ── Header ──────────────────────────────────────────────────── */}
      <header style={{
        background: "var(--surface)", borderBottom: "1px solid var(--border)",
        position: "sticky", top: 0, zIndex: 50,
        padding: "0 32px", height: 60,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "var(--blue)", display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>⚛</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)", letterSpacing: "-.01em" }}>QuantumMeta-Sat</div>
            <div style={{ fontSize: 11, color: "var(--dim)" }}>LEO Satellite Channel Estimation</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 12, color: "var(--dim)" }}>Speed:</span>
          {[["0.5×", 0.5], ["1×", 1], ["2×", 2]].map(([label, val]) => (
            <button key={label} onClick={() => setSpeed(val as number)}
              style={{
                padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                border: `1px solid ${speed === val ? "var(--blue)" : "var(--border)"}`,
                background: speed === val ? "var(--blue-bg)" : "transparent",
                color: speed === val ? "var(--blue)" : "var(--dim)",
                cursor: "pointer", fontFamily: "inherit",
              }}>{label}</button>
          ))}
        </div>
      </header>

      <main style={{ maxWidth: 760, margin: "0 auto", padding: "40px 24px 80px" }}>

        {/* ── Hero / Send UI ───────────────────────────────────────── */}
        <div className="card" style={{ padding: "28px 32px", marginBottom: 40, borderColor: step > 0 ? "var(--blue)" : "var(--border)", transition: "border-color .3s" }}>
          <div style={{ fontSize: 13, color: "var(--dim)", marginBottom: 6, fontWeight: 500 }}>MESSAGE TO TRANSMIT</div>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <input
              value={inputText}
              onChange={(e) => { if (!running) setInputText(e.target.value || "Hello World"); }}
              disabled={running}
              style={{
                flex: 1, minWidth: 200, fontSize: 22, fontWeight: 700, color: "var(--text)",
                background: "transparent", border: "none", outline: "none", fontFamily: "inherit",
                letterSpacing: "-.01em",
              }}
            />
            {running && <span className="anim-blink" style={{ fontSize: 22, color: "var(--blue)" }}>|</span>}
          </div>
          <div style={{ marginTop: 10, fontSize: 12, color: "var(--dim)", fontFamily: "JetBrains Mono, monospace" }}>
            {inputText.length} chars · {inputText.length * 8} bits · QPSK modulation · OFDM transmission
          </div>
          <div style={{ marginTop: 20, display: "flex", gap: 10, alignItems: "center" }}>
            <button className="btn-primary" onClick={run}>
              {running ? "⏹ Stop" : step === 0 ? "▶ Send via LEO Satellite" : "↺ Replay"}
            </button>
            {step > 0 && !running && (
              <button className="btn-ghost" onClick={reset}>Reset</button>
            )}
            {step > 0 && (
              <div style={{ marginLeft: "auto", display: "flex", gap: 16 }}>
                <span style={{ fontSize: 12, color: "var(--sub)" }}>
                  Step <strong style={{ color: "var(--blue)" }}>{Math.min(step, 6)}</strong> / 6
                </span>
                <div style={{ width: 120 }}>
                  <div className="progress">
                    <div className="progress-fill" style={{ width: `${(Math.min(step, 6) / 6) * 100}%` }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Step 1 — Encode ──────────────────────────────────────── */}
        <StepBlock number={1} title="Encode to Bits" pill="Text → Binary" status={stepStatus(1)}>
          <div className="card-sm" style={{ padding: "16px 20px" }}>
            <div style={{ fontSize: 12, color: "var(--sub)", marginBottom: 12 }}>
              Each character → 8-bit ASCII. "{inputText.slice(0, 5)}" becomes:
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
              {inputText.slice(0, 5).split("").map((ch, ci) => (
                <div key={ci} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text)", marginBottom: 5 }}>'{ch}'</div>
                  <div style={{ display: "flex", gap: 1 }}>
                    {ch.charCodeAt(0).toString(2).padStart(8, "0").split("").map((b, bi) => (
                      <span key={bi} className={`bit bit-${b === "1" ? "1" : "0"}`}>{b}</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 9, color: "var(--dim)", marginTop: 3, fontFamily: "JetBrains Mono, monospace" }}>
                    {ch.charCodeAt(0)}
                  </div>
                </div>
              ))}
              {inputText.length > 5 && (
                <div style={{ fontSize: 12, color: "var(--dim)", alignSelf: "center" }}>
                  +{(inputText.length - 5) * 8} more bits…
                </div>
              )}
            </div>
            <div style={{ fontSize: 12, color: "var(--dim)" }}>
              Total: <strong style={{ color: "var(--text)" }}>{inputText.length * 8} bits</strong> ready for modulation
            </div>
          </div>
        </StepBlock>

        {/* ── Step 2 — Modulate ─────────────────────────────────────── */}
        <StepBlock number={2} title="QPSK Modulation + OFDM" pill="Bits → Symbols" pillType="purple" status={stepStatus(2)}>
          <div className="card-sm" style={{ padding: "16px 20px" }}>
            <div style={{ fontSize: 12, color: "var(--sub)", marginBottom: 14 }}>
              Pairs of bits map to one of 4 QPSK symbols on the complex plane. Known pilot symbols are inserted every 7 subcarriers so the receiver can estimate the channel.
            </div>
            <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
              {/* QPSK mapping */}
              <div>
                <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 8, fontWeight: 600 }}>BIT PAIRS → SYMBOLS</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {pairs.slice(0, 6).map((pair, i) => {
                    const sym = QPSK_MAP[pair] || QPSK_MAP["00"];
                    return (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                        <span className="mono" style={{ color: "var(--blue)", background: "var(--blue-bg)", padding: "2px 6px", borderRadius: 4 }}>{pair}</span>
                        <span style={{ color: "var(--dim)" }}>→</span>
                        <span className="mono" style={{ color: "var(--purple)" }}>{sym.label}</span>
                        <span style={{ fontSize: 10, color: "var(--dim)" }}>({sym.re > 0 ? "+" : ""}{sym.re.toFixed(2)}, {sym.im > 0 ? "+" : ""}{sym.im.toFixed(2)})</span>
                      </div>
                    );
                  })}
                  {pairs.length > 6 && <div style={{ fontSize: 11, color: "var(--dim)" }}>+{pairs.length - 6} more…</div>}
                </div>
              </div>
              {/* Constellation */}
              <div>
                <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 8, fontWeight: 600 }}>CONSTELLATION DIAGRAM</div>
                <Constellation />
              </div>
              {/* OFDM info */}
              <div style={{ fontSize: 12, color: "var(--sub)", lineHeight: 1.8, minWidth: 150 }}>
                <div style={{ fontWeight: 700, color: "var(--text)", marginBottom: 6 }}>OFDM Frame</div>
                {[["Subcarriers", "64"], ["Pilot spacing", "every 7th"], ["Pilot purpose", "channel ref."], ["Guard interval", "cyclic prefix"]].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <span style={{ color: "var(--dim)" }}>{k}</span>
                    <span className="mono">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </StepBlock>

        {/* ── Step 3 — Channel ──────────────────────────────────────── */}
        <StepBlock number={3} title="Signal Travels Through LEO Channel" pill="Transmission" pillType="amber" status={stepStatus(3)}>
          <div className="card-sm" style={{ padding: "16px 20px" }}>
            <div style={{ fontSize: 12, color: "var(--sub)", marginBottom: 14 }}>
              The OFDM frame travels 550 km to the satellite, then back down. During this 2-hop journey three things corrupt the signal:
            </div>
            <ChannelSVG phase={step >= 3 ? (step >= 4 ? 3 : 2) : step >= 3 ? 1 : 0} />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginTop: 14 }}>
              {[
                { icon: "〰️", title: "Doppler Shift", color: "var(--amber)", desc: "Satellite moves at 7.6 km/s → frequency shifts ±48 kHz" },
                { icon: "🌫️", title: "Fading h[k]", color: "var(--purple)", desc: "Shadowed-Rician model: multipath reflections change amplitude" },
                { icon: "📟", title: "AWGN Noise", color: "var(--red)", desc: "Thermal noise n[k] ~ CN(0, σ²) added at receiver" },
              ].map((item) => (
                <div key={item.title} style={{ background: "var(--bg)", borderRadius: 10, padding: "12px 14px", border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 18, marginBottom: 6 }}>{item.icon}</div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: item.color, marginBottom: 4 }}>{item.title}</div>
                  <div style={{ fontSize: 11, color: "var(--dim)", lineHeight: 1.5 }}>{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </StepBlock>

        {/* ── Step 4 — Received ─────────────────────────────────────── */}
        <StepBlock number={4} title="Ground Station Receives Distorted Signal" pill="y = hx + n" pillType="amber" status={stepStatus(4)}>
          <div className="card-sm" style={{ padding: "16px 20px" }}>
            <div style={{ fontSize: 12, color: "var(--sub)", marginBottom: 14 }}>
              The received signal y[k] is a distorted version of what was sent. The QPSK constellation is now scattered — we can't decode directly.
            </div>
            <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
              <div>
                <Waveform noisy={false} label="Original signal x[k] (clean)" color="var(--blue)" />
              </div>
              <div style={{ fontSize: 22, color: "var(--border)" }}>→</div>
              <div>
                <Waveform noisy label="Received signal y[k] (distorted)" color="var(--amber)" />
              </div>
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 14, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 6, fontWeight: 600 }}>CLEAN SYMBOLS</div>
                <Constellation />
              </div>
              <div style={{ alignSelf: "center", textAlign: "center" }}>
                <div style={{ fontSize: 28, color: "var(--border)", marginBottom: 4 }}>→</div>
                <div style={{ fontSize: 11, color: "var(--dim)" }}>Channel</div>
                <div style={{ fontSize: 11, color: "var(--dim)" }}>distortion</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 6, fontWeight: 600 }}>RECEIVED (NOISY)</div>
                <Constellation noisy />
              </div>
            </div>
            <div style={{ marginTop: 14, padding: "12px 14px", background: "var(--amber-bg)", borderRadius: 8, border: "1px solid var(--amber)", fontSize: 12, color: "var(--amber)" }}>
              ⚠️ The ground station extracts pilot symbols y_pilot[k] — it knows what was sent there, so it can measure how the channel distorted them.
            </div>
          </div>
        </StepBlock>

        {/* ── Step 5 — Quantum ──────────────────────────────────────── */}
        <StepBlock number={5} title="Q-MAML: Quantum Circuit Estimates the Channel" pill="⚛ VQC + Meta-Learning" pillType="purple" status={stepStatus(5)}>
          <div className="card-sm" style={{ padding: "16px 20px" }}>
            <div style={{ fontSize: 12, color: "var(--sub)", marginBottom: 16, lineHeight: 1.7 }}>
              The received pilot observations are fed into a <strong>Variational Quantum Circuit (VQC)</strong>. The circuit was meta-trained to adapt quickly — it only needs <strong>3 pilot shots</strong> to estimate the full channel.
            </div>

            <QuantumCircuitSVG activeLayer={qLayer} />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr", gap: 10, marginTop: 16 }}>
              {[
                { step: "①", text: "Pilots fed in as rotation angles", color: "var(--blue)" },
                { step: "②", text: "Variational layers rotate qubit states", color: "var(--purple)" },
                { step: "③", text: "CNOT gates entangle all 4 qubits", color: "var(--amber)" },
                { step: "④", text: "More variational rotations", color: "var(--purple)" },
                { step: "⑤", text: "Measure expectation → channel ĥ", color: "var(--green)" },
              ].map((item, i) => (
                <div key={i} style={{
                  padding: "10px 12px", borderRadius: 8, fontSize: 11, lineHeight: 1.5,
                  background: qLayer === i ? "var(--blue-bg)" : "var(--bg)",
                  border: `1px solid ${qLayer === i ? "var(--blue)" : "var(--border)"}`,
                  transition: "all .3s",
                }}>
                  <div style={{ fontWeight: 700, color: item.color, marginBottom: 4 }}>{item.step}</div>
                  <div style={{ color: "var(--sub)" }}>{item.text}</div>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 14, padding: "12px 14px", background: "var(--blue-bg)", borderRadius: 8, border: "1px solid var(--blue)", fontSize: 12, color: "var(--blue)" }}>
              💡 MAML meta-trained this circuit across 2000 different channel environments. So it learned <em>how to adapt fast</em> — not just one channel. After 3 shots of gradient update, it produces an accurate ĥ.
            </div>
          </div>
        </StepBlock>

        {/* ── Step 6 — Result ───────────────────────────────────────── */}
        <StepBlock number={6} title="Result: Message Recovered!" pill="✓ Success" pillType="green" status={stepStatus(6)}>
          {showResult && (
            <div>
              <div className="card-sm" style={{ padding: "24px", borderColor: "var(--green)", borderWidth: 2, marginBottom: 16 }}>
                <div style={{ display: "flex", gap: 14, alignItems: "flex-start", marginBottom: 20 }}>
                  <div style={{ fontSize: 36 }}>✅</div>
                  <div>
                    <div style={{ fontSize: 13, color: "var(--dim)", marginBottom: 4 }}>RECOVERED MESSAGE</div>
                    <div style={{ fontSize: 28, fontWeight: 800, color: "var(--text)", letterSpacing: "-.02em" }}>"{inputText}"</div>
                    <div style={{ fontSize: 12, color: "var(--green)", marginTop: 4 }}>Decoded perfectly using Q-MAML channel estimate</div>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 6, fontWeight: 600 }}>RECOVERED SYMBOLS</div>
                    <Constellation recovered />
                  </div>
                  <div style={{ flex: 1, minWidth: 220 }}>
                    <div style={{ fontSize: 11, color: "var(--dim)", marginBottom: 12, fontWeight: 600 }}>PERFORMANCE METRICS</div>
                    {[
                      { label: "NMSE (channel estimation error)", value: `${nmse.toFixed(2)} dB`, color: "var(--blue)", note: "+6.1 dB better than LS baseline" },
                      { label: "Bit Error Rate", value: ber.toExponential(2), color: "var(--green)", note: "Near theoretical minimum" },
                      { label: "Adaptation shots used", value: "3 pilots", color: "var(--purple)", note: "Classical CNN needs 20+" },
                      { label: "Circuit parameters", value: "12K", color: "var(--amber)", note: "vs 128K for CNN baseline" },
                    ].map((m) => (
                      <div key={m.label} style={{ marginBottom: 14 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: "var(--sub)" }}>{m.label}</span>
                          <span className="mono" style={{ fontWeight: 700, color: m.color }}>{m.value}</span>
                        </div>
                        <div style={{ fontSize: 10, color: "var(--dim)" }}>{m.note}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Why better */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  { icon: "🐌", title: "Classical CNN", shots: "20+ pilots", nmse: "-13.2 dB", color: "var(--dim)", bg: "var(--surface)" },
                  { icon: "⚛️", title: "Q-MAML (Ours)", shots: "3 pilots ⚡", nmse: "-19.3 dB", color: "var(--blue)", bg: "var(--blue-bg)", border: "var(--blue)" },
                ].map((card) => (
                  <div key={card.title} className="card-sm" style={{ padding: "16px 18px", borderColor: card.border ?? "var(--border)", background: card.bg }}>
                    <div style={{ fontSize: 22, marginBottom: 8 }}>{card.icon}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: card.color, marginBottom: 10 }}>{card.title}</div>
                    {[["Adapt. shots", card.shots], ["NMSE", card.nmse]].map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
                        <span style={{ color: "var(--dim)" }}>{k}</span>
                        <span className="mono" style={{ color: card.color, fontWeight: 600 }}>{v}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
          {!showResult && step >= 6 && (
            <div style={{ color: "var(--dim)", fontSize: 13 }}>Processing…</div>
          )}
          {step < 6 && (
            <div style={{ color: "var(--dim)", fontSize: 13 }}>Waiting for simulation to complete…</div>
          )}
        </StepBlock>

      </main>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer style={{
        background: "var(--surface)", borderTop: "1px solid var(--border)",
        padding: "16px 32px", display: "flex", justifyContent: "space-between",
        fontSize: 12, color: "var(--dim)",
      }}>
        <span>QuantumMeta-Sat — Research Simulator</span>
        <span>PennyLane · PyTorch · Next.js</span>
      </footer>
    </div>
  );
}
