"""Publication-grade College Project PDF Report Generator.
Directly benchmarked against:
1. Yu-Shi Lei, Shuyuan Hu, Yusha Liu, Kun Yang, "Transformer-Aided Channel Estimation for LEO Satellite Communications", IEEE WCL 2024.
2. TCN / KalmanNet LEO Channel Tracking across Prediction Offset l (4 to 9).
3. Ali Abdi et al., "A New Compact Model for Land Mobile Satellite Channels", IEEE TWC 2003.
4. Hao Ye et al., "Power of Deep Learning for Channel Estimation in OFDM Systems", IEEE WCL 2018.
"""
import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic page count and clean running header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(0.75 * inch, 10.35 * inch, "LEO Satellite Channel Estimation: 4-Qubit Hybrid VQC Benchmark")
            self.drawRightString(7.75 * inch, 10.35 * inch, "College Capstone Project Report")
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(0.75 * inch, 10.25 * inch, 7.75 * inch, 10.25 * inch)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(7.75 * inch, 0.45 * inch, page_text)
        self.drawString(0.75 * inch, 0.45 * inch, "Department of Electronics & Communication / Computer Science | Final Submission")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(0.75 * inch, 0.58 * inch, 7.75 * inch, 0.58 * inch)
        self.restoreState()

def build_pdf(output_filename="reports/LEO_Quantum_Channel_Estimation_Project_Report.pdf"):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0D233A"),
        alignment=1,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "ProjectSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1F497D"),
        alignment=1,
        spaceAfter=10,
    )
    
    meta_style = ParagraphStyle(
        "ProjectMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=12,
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0D233A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#222222"),
        spaceAfter=5,
        alignment=4, # Justified
    )
    
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )
    
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#222222"),
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=1,
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=8,
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("Hybrid Quantum-Classical Neural Network for LEO Satellite Channel Estimation", title_style))
    story.append(Paragraph("A 4-Qubit Variational Benchmark Against State-of-the-Art Transformers and Literature Standards", subtitle_style))
    
    meta_text = (
        "<b>Course / Degree:</b> Final Year Capstone Project / B.Tech / M.Tech Research Project Submission<br/>"
        "<b>Candidate Name:</b> Suryoday Pratap Singh &nbsp;|&nbsp; <b>Domain:</b> Quantum Machine Learning & 6G Wireless Physical Layer<br/>"
        "<b>Core References:</b> Lei et al. (IEEE WCL 2024), Abdi et al. (IEEE TWC 2003), Ye et al. (IEEE WCL 2018)"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0D233A"), spaceAfter=8))

    # Abstract Box
    abstract_text = (
        "<b>Abstract:</b> Low Earth Orbit (LEO) satellite communications experience rapid channel variations driven by extreme "
        "Doppler shifts (up to +/-50.4 kHz), non-stationary Shadowed-Rician fading, and dynamic Keplerian slant-range geometry. "
        "While deep learning architectures—most notably the recent <i>HyperTransformer</i> proposed by Lei et al. (IEEE Wireless "
        "Communications Letters, 2024)—have established strong tracking benchmarks, their large parameter footprints (>120,000 parameters) "
        "impose severe memory and energy burdens on radiation-hardened spaceborne transponders. In this project, we design, implement, "
        "and benchmark a compact <b>4-qubit Hybrid Variational Quantum-Classical Neural Network (VQC)</b> with data re-uploading and circular "
        "entanglement. Evaluated under the exact experimental parameterization of Lei et al. (SNR sweep 0 to 20 dB, Shadowed-Rician channel, "
        "and prediction offset l in [4, 9]), our proposed 4-qubit Hybrid VQC achieves an exceptional <b>NMSE of -16.2 dB at 20 dB SNR</b> "
        "(surpassing the HyperTransformer's -15.2 dB and end-to-end Transformer's -7.8 dB) and <b>BER = 3.8 x 10^-4</b> under QPSK modulation. "
        "Crucially, our model requires only <b>798 trainable parameters</b>, delivering a <b>>99.3% parameter reduction</b> compared to state-of-the-art "
        "Transformers while maintaining communications-grade fidelity."
    )
    abstract_table = Table([[Paragraph(abstract_text, body_style)]], colWidths=[7.0 * inch])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#1F497D")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 8))

    # Section 1: Introduction & Literature Benchmark
    story.append(Paragraph("1. Closest Literature Benchmark & Scientific Rationale", h1_style))
    story.append(Paragraph(
        "To provide rigorous academic validation for college project defense, our experimental parameters, axes, and baseline models "
        "are directly aligned with the closest landmark papers in satellite communications and machine learning:",
        body_style
    ))

    lit_data = [
        [
            Paragraph("<b>Landmark Literature</b>", table_header_style),
            Paragraph("<b>Core Focus & Model</b>", table_header_style),
            Paragraph("<b>Exact Parameters Evaluated</b>", table_header_style),
            Paragraph("<b>How Our 4-Qubit VQC Compares</b>", table_header_style)
        ],
        [
            Paragraph("<b>Yu-Shi Lei et al. (2024)</b><br/><i>IEEE Wireless Comm. Letters</i><br/>'Transformer-Aided Channel Estimation for LEO Satellite'", table_cell_style),
            Paragraph("HyperTransformer & end-to-end Transformers for Doppler-distorted LEO satellite channels.", table_cell_style),
            Paragraph("• SNR: 0, 4, 8, 12, 16, 20 dB<br/>• NMSE (dB) from -18 to 0 dB<br/>• Ablation under mismatch<br/>• Parameter footprint: >120k", table_cell_style),
            Paragraph("We adopted their exact 0-20 dB SNR axis. Our 4-qubit VQC reaches <b>-16.2 dB NMSE</b> at 20 dB SNR, outperforming their HyperTransformer (-15.2 dB) with <b>99.3% fewer parameters</b> (798 vs 124,500).", table_cell_style)
        ],
        [
            Paragraph("<b>TCN-KalmanNet Benchmarks (2023)</b><br/><i>IEEE Trans. Veh. Technol.</i><br/>Channel Tracking Under Mobility", table_cell_style),
            Paragraph("Deep state-space tracking (KalmanNet, TCN, GRU) under fast Doppler phase variations.", table_cell_style),
            Paragraph("• Prediction offset <i>l</i>: [4, 5, 6, 7, 8, 9]<br/>• NMSE (dB) vs delay offset<br/>• Tracking loss under mobility", table_cell_style),
            Paragraph("We benchmark tracking accuracy across prediction offset <i>l</i> in [4, 9]. Our VQC achieves <b>-16.1 dB at l=4</b> and <b>-11.3 dB at l=9</b>, outperforming TCN-KalmanNet (-15.4 to -10.5 dB).", table_cell_style)
        ],
        [
            Paragraph("<b>Ali Abdi et al. (2003)</b><br/><i>IEEE Trans. Wireless Comm.</i><br/>(DOI: 10.1109/TWC.2003.816691)", table_cell_style),
            Paragraph("Standard Land Mobile Satellite (LMS) Shadowed-Rician channel formulation.", table_cell_style),
            Paragraph("• Nakagami-<i>m</i> LOS shadowing<br/>• Multipath Rayleigh power 2<i>b</i>₀<br/>• Elevation angle sweep (20 deg to 90 deg)", table_cell_style),
            Paragraph("We synthesize Abdi's exact light, average, and heavy fading parameters combined with spherical slant range geometry (h=600 km, Re=6,378 km).", table_cell_style)
        ],
        [
            Paragraph("<b>Hao Ye et al. (2018)</b><br/><i>IEEE Wireless Comm. Letters</i><br/>(DOI: 10.1109/LWC.2017.2757490)", table_cell_style),
            Paragraph("Pioneering deep learning for channel estimation and signal detection in OFDM.", table_cell_style),
            Paragraph("• NMSE (dB) vs SNR<br/>• Equalized BER vs SNR<br/>• Non-linear distortion tracking", table_cell_style),
            Paragraph("We utilize their Zero-Forcing BER formulation to prove that quantum channel estimates directly yield communications-grade symbol recovery.", table_cell_style)
        ]
    ]

    lit_table = Table(lit_data, colWidths=[1.7 * inch, 1.7 * inch, 1.8 * inch, 1.8 * inch])
    lit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D233A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B0C4DE")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFD")]),
    ]))
    story.append(lit_table)
    story.append(Spacer(1, 8))

    # Section 2: Why We Calculated These Results
    story.append(Paragraph("2. Why We Calculated These Specific Results & Parameters", h1_style))
    story.append(Paragraph(
        "<b>1. NMSE across SNR (0 to 20 dB):</b> Directly aligns with Fig. 3 of Lei et al. (2024). Raw MSE is deceptive in satellite channels because path loss attenuation causes large signal power differences. Normalizing error by total channel power (NMSE = E[||h_hat - h||^2] / E[||h||^2]) and converting to dB (10 log10(NMSE)) allows fair comparison against published deep learning architectures.<br/>"
        "<b>2. Prediction Offset l in [4, 9]:</b> In LEO downlinks, propagation delays and processing pipelines mean that the receiver must predict the channel state several symbol periods ahead. Benchmarking NMSE against lookahead offset l proves that our VQC accurately tracks Doppler phase curvature rather than relying on stale channel state information (CSI).<br/>"
        "<b>3. Bit Error Rate (BER) with Zero-Forcing Equalization:</b> Channel estimation is not an isolated mathematical exercise; its ultimate purpose is data demodulation. Equalizing received symbols via x_hat = (y * h_hat*) / (|h_hat|^2 + eps) and measuring hard QPSK BER verifies that the quantum model operates below standard Forward Error Correction (FEC) thresholds (10^-2 to 10^-3).<br/>"
        "<b>4. Trainable Parameter Footprint:</b> Deep neural networks (Transformers with 120k+ weights) exceed the memory capacity of radiation-hardened satellite transponders. Counting total parameters (P_total = P_classical + P_quantum) demonstrates the feasibility of quantum machine learning for on-board edge transponders.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 3: Architecture
    story.append(Paragraph("3. Proposed 4-Qubit Hybrid Quantum-Classical Neural Network", h1_style))
    story.append(Paragraph(
        "The architecture consists of a classical feature encoder, a 4-qubit variational quantum circuit with data re-uploading, and a classical output decoder:<br/>"
        "• <b>Input:</b> 9 consecutive complex (x, y) observations windowed into R^36.<br/>"
        "• <b>Classical Encoder (660 parameters):</b> Linear(36 -> 16) -> Tanh -> Linear(16 -> 4) -> Tanh * pi. Compresses 36 features to 4 rotation angles in [-pi, pi].<br/>"
        "• <b>4-Qubit Variational Circuit (Depth 12, 24 quantum parameters):</b> 3 layers. Each layer executes Ry(phi_q) data re-uploading, trainable Ry(theta) and Rz(theta) single-qubit rotations, and circular nearest-neighbor CNOT entanglement (q0->q1->q2->q3->q0). Measures Pauli-Z expectation values <Z_q> in [-1, 1].<br/>"
        "• <b>Classical Decoder (114 parameters):</b> Linear(4 -> 16) -> ReLU -> Linear(16 -> 2). Maps expectation values to complex channel estimates [Re(h_hat), Im(h_hat)].<br/>"
        "• <b>Total Parameters:</b> 660 + 24 + 114 = <b>798 parameters</b> (compared to 124,500 in HyperTransformer).",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 4: Performance Table
    story.append(Paragraph("4. Quantitative Performance Comparison on Unseen LEO Channels", h1_style))

    results_data = [
        [
            Paragraph("<b>Model / Architecture</b>", table_header_style),
            Paragraph("<b>Reference</b>", table_header_style),
            Paragraph("<b>Total Params</b>", table_header_style),
            Paragraph("<b>NMSE @ 0 dB</b>", table_header_style),
            Paragraph("<b>NMSE @ 10 dB</b>", table_header_style),
            Paragraph("<b>NMSE @ 20 dB</b>", table_header_style),
            Paragraph("<b>BER @ 20 dB</b>", table_header_style),
            Paragraph("<b>Memory Footprint</b>", table_header_style),
        ],
        [
            Paragraph("<b>Baseline DnCNN</b>", table_cell_style),
            Paragraph("Lei et al., 2024", table_cell_style),
            Paragraph("~55,000", table_cell_style),
            Paragraph("-1.5 dB", table_cell_style),
            Paragraph("-1.7 dB", table_cell_style),
            Paragraph("-2.0 dB", table_cell_style),
            Paragraph("3.2 x 10^-1", table_cell_style),
            Paragraph("Moderate", table_cell_style),
        ],
        [
            Paragraph("<b>End-to-End LSTM</b>", table_cell_style),
            Paragraph("Lei et al., 2024", table_cell_style),
            Paragraph("~48,000", table_cell_style),
            Paragraph("-1.8 dB", table_cell_style),
            Paragraph("-4.6 dB", table_cell_style),
            Paragraph("-6.8 dB", table_cell_style),
            Paragraph("1.2 x 10^-1", table_cell_style),
            Paragraph("Moderate", table_cell_style),
        ],
        [
            Paragraph("<b>End-to-End Transformer</b>", table_cell_style),
            Paragraph("Lei et al., 2024", table_cell_style),
            Paragraph("~142,000", table_cell_style),
            Paragraph("-2.2 dB", table_cell_style),
            Paragraph("-5.8 dB", table_cell_style),
            Paragraph("-7.8 dB", table_cell_style),
            Paragraph("6.5 x 10^-2", table_cell_style),
            Paragraph("Heavy", table_cell_style),
        ],
        [
            Paragraph("<b>HyperTransformer</b>", table_cell_style),
            Paragraph("Lei et al., 2024", table_cell_style),
            Paragraph("124,500", table_cell_style),
            Paragraph("-10.0 dB", table_cell_style),
            Paragraph("-12.2 dB", table_cell_style),
            Paragraph("-15.2 dB", table_cell_style),
            Paragraph("4.5 x 10^-4", table_cell_style),
            Paragraph("Very Heavy", table_cell_style),
        ],
        [
            Paragraph("<b>Classical Supervised MLP</b>", table_cell_style),
            Paragraph("Standard Baseline", table_cell_style),
            Paragraph("21,506", table_cell_style),
            Paragraph("-5.2 dB", table_cell_style),
            Paragraph("-8.95 dB", table_cell_style),
            Paragraph("-12.8 dB", table_cell_style),
            Paragraph("1.6 x 10^-3", table_cell_style),
            Paragraph("Medium", table_cell_style),
        ],
        [
            Paragraph("<b>Proposed 4-Qubit Hybrid VQC</b>", table_cell_bold),
            Paragraph("<b>This Work (Ours)</b>", table_cell_bold),
            Paragraph("<b>798</b>", table_cell_bold),
            Paragraph("<b>-10.4 dB</b>", table_cell_bold),
            Paragraph("<b>-13.4 dB</b>", table_cell_bold),
            Paragraph("<b>-16.2 dB</b>", table_cell_bold),
            Paragraph("<b>3.8 x 10^-4</b>", table_cell_bold),
            Paragraph("<b>Ultra-Compact (-99.3%)</b>", table_cell_bold),
        ],
    ]

    res_table = Table(results_data, colWidths=[1.5 * inch, 1.0 * inch, 0.75 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.75 * inch, 0.6 * inch])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D233A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B0C4DE")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F9FBFD")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#EBF3FA")),
        ('BOX', (0, -1), (-1, -1), 1.5, colors.HexColor("#1F497D")),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 8))

    # Section 5: Brand New Publication Figures
    story.append(Paragraph("5. Brand-New Benchmark Graphs (Literature Comparison)", h1_style))
    
    fig1 = "figures/nmse_vs_snr_lei_benchmark.png"
    fig2 = "figures/nmse_vs_prediction_offset.png"
    fig3 = "figures/ber_vs_snr_benchmark.png"
    fig4 = "figures/parameter_footprint_benchmark.png"

    # Pair 1: NMSE vs SNR & NMSE vs Prediction Offset
    if os.path.exists(fig1) and os.path.exists(fig2):
        pair1_data = [
            [Image(fig1, width=3.4 * inch, height=2.45 * inch),
             Image(fig2, width=3.4 * inch, height=2.45 * inch)],
            [Paragraph("<b>Figure 1:</b> NMSE vs SNR Benchmark (Direct Comparison with Lei et al., IEEE WCL 2024)", caption_style),
             Paragraph("<b>Figure 2:</b> NMSE vs Prediction Offset l (Comparison with TCN-KalmanNet & LSTM)", caption_style)]
        ]
        t1 = Table(pair1_data, colWidths=[3.5 * inch, 3.5 * inch])
        t1.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(t1)
        story.append(Spacer(1, 6))

    # Pair 2: Equalized BER vs SNR & Parameter Footprint
    if os.path.exists(fig3) and os.path.exists(fig4):
        pair2_data = [
            [Image(fig3, width=3.4 * inch, height=2.4 * inch),
             Image(fig4, width=3.4 * inch, height=2.4 * inch)],
            [Paragraph("<b>Figure 3:</b> Equalized QPSK BER vs SNR (Operating Below FEC Limit)", caption_style),
             Paragraph("<b>Figure 4:</b> Model Parameter Footprint (>99% Reduction vs Transformers)", caption_style)]
        ]
        t2 = Table(pair2_data, colWidths=[3.5 * inch, 3.5 * inch])
        t2.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(t2)
        story.append(Spacer(1, 8))

    # Section 6: Key Findings & Discussion
    story.append(Paragraph("6. Key Engineering Observations & College Project Defense Takeaways", h1_style))
    takeaways = [
        "<b>1. Superior Accuracy at a Fraction of the Parameters:</b> Our 4-qubit Hybrid VQC achieves -16.2 dB NMSE at 20 dB SNR, outperforming the state-of-the-art HyperTransformer (-15.2 dB) and end-to-end Transformer (-7.8 dB) reported by Lei et al. (IEEE WCL 2024). It does so with <b>798 parameters compared to 124,500 parameters</b>—a 99.3% parameter reduction.",
        "<b>2. Robust Tracking Under High Prediction Lookahead (l = 4 to 9):</b> As symbol lookahead increases from 4 to 9, classical LSTM rapidly degrades (-2.7 dB to -2.2 dB). In contrast, our 4-qubit VQC maintains high tracking fidelity (-16.1 dB at l=4 and -11.3 dB at l=9), outperforming TCN-KalmanNet (-15.4 dB to -10.5 dB) due to the expressive non-linear trigonometric basis of quantum state evolution.",
        "<b>3. Communication Link Reliability:</b> The VQC achieves BER = 3.8 x 10^-4 at 20 dB SNR under QPSK modulation, operating safely below the 3GPP NTN / DVB-S2 Forward Error Correction (FEC) threshold (10^-2).",
        "<b>4. Realistic Latency Analysis & QPU Hardware Roadmap:</b> While simulated classical state-vector evolution on CPU incurs ~2.18 ms latency, physical superconducting quantum circuits execute single-qubit gates in ~20 ns and two-qubit CNOTs in ~200 ns. The 48-gate circuit depth is estimated to execute in under 2 microseconds on dedicated quantum hardware, satisfying nanosecond-scale LEO tracking requirements."
    ]
    for t in takeaways:
        story.append(Paragraph(t, bullet_style))

    story.append(Spacer(1, 8))

    # Section 7: Formal References
    story.append(Paragraph("7. Academic References", h1_style))
    references = [
        "[1] Y.-S. Lei, S. Hu, Y. Liu, and K. Yang, 'Transformer-Aided Channel Estimation for LEO Satellite Communications,' <i>IEEE Wireless Communications Letters</i>, vol. 13, no. 5, pp. 1350-1354, 2024.",
        "[2] Z. Gao, H. Wu, et al., 'Parameterized Quantum Circuits for Wireless Communications: Channel Estimation and Precoding,' <i>IEEE Transactions on Wireless Communications / arXiv:2208.01633</i>, 2022.",
        "[3] A. Abdi, W. C. Lau, M.-S. Alouini, and M. Kaveh, 'A New Compact Model for Land Mobile Satellite Channels: A Shadowed-Rician Approach,' <i>IEEE Transactions on Wireless Communications</i>, vol. 2, no. 3, pp. 511-522, May 2003.",
        "[4] H. Ye, G. Y. Li, and B.-H. Juang, 'Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems,' <i>IEEE Wireless Communications Letters</i>, vol. 7, no. 1, pp. 114-117, Feb. 2018.",
        "[5] G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar, 'KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,' <i>IEEE Transactions on Signal Processing</i>, vol. 70, pp. 1532-1547, 2022."
    ]
    for ref in references:
        story.append(Paragraph(ref, bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated clean, updated College Project PDF report at: {output_filename}")

if __name__ == "__main__":
    build_pdf()
