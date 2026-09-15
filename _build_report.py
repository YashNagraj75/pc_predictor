"""Build the progress report PDF (<= 6 pages)."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, KeepTogether)

TITLE = "#111111"
ACC = "#1b6ca8"
GREY = "#555555"

body = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.6, leading=13.0,
                      spaceAfter=6, textColor=TITLE, alignment=4)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                    spaceBefore=11, spaceAfter=5, textColor=ACC)
h0 = ParagraphStyle("h0", fontName="Helvetica-Bold", fontSize=16, leading=19,
                    spaceAfter=3, textColor=TITLE)
sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=9.4, leading=12.4,
                     spaceAfter=10, textColor=GREY)
cap = ParagraphStyle("cap", fontName="Times-Italic", fontSize=8.3, leading=10.6,
                     spaceBefore=3, spaceAfter=9, textColor=GREY)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=3,
                        spaceAfter=3.5)
q = ParagraphStyle("q", parent=body, leftIndent=13, bulletIndent=3, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8.6, leading=11,
                       textColor=GREY)

S = []
A = S.append


def para(t, st=body):
    A(Paragraph(t, st))


def bullets(items, st=bullet):
    for it in items:
        A(Paragraph(it, st, bulletText="\u2022"))


def tbl(data, widths, align_right=()):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4),
        ("FONT", (0, 1), (-1, -1), "Times-Roman", 8.8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(ACC)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(ACC)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------- title
para("Predictive Coding Predictors for Joint-Embedding World Models", h0)
para("Progress report and request for feedback &nbsp;|&nbsp; Yash Nagraj "
     "&nbsp;|&nbsp; August 2026", sub)

para("<b>Where this stands.</b> The project began as a proposal to train the "
     "predictor of an I-JEPA-style joint-embedding world model with predictive "
     "coding (PC) instead of backpropagation (BP). The original motivation was "
     "that PC might supply a better gradient signal. I now think that "
     "motivation was wrong, and I have replaced it with a different one. This "
     "report covers what I built, what I measured, what changed my mind, and "
     "three questions I would like your opinion on. Everything numerical below "
     "is from code I wrote and ran; nothing is quoted from a paper.")

# ---------------------------------------------------------------- 1
para("1. The reframing: PC does not give better gradients", h1)
para("Millidge, Tschantz and Buckley (2020) show that PC approximates BP along "
     "arbitrary computation graphs. So &ldquo;PC gives a better gradient than "
     "BP&rdquo; is close to self-defeating: to the extent PC works, it works "
     "<i>because</i> it recovers the BP gradient. I verified this myself rather "
     "than take it on faith &mdash; I implemented PC over an explicit computation "
     "graph in JAX and compared it against autodiff on three topologies (a plain "
     "chain, a residual network with folded skips, and one with explicit skips). "
     "The PC and BP weight gradients agree to <b>cosine similarity 1.000000</b> "
     "on all three. On a single branch the PC gradient is "
     "<b>&minus;11.370010</b> against BP&rsquo;s <b>&minus;11.370533</b>.")
para("Two things fell out of that exercise that I did not expect, and that I "
     "think are the more useful results:")
bullets([
    "<b>The equivalence is load-bearing on one assumption.</b> Dropping the "
    "fixed-prediction assumption changes that same gradient from "
    "&minus;11.370010 to <b>&minus;3.413946</b> &mdash; a factor of three, not a "
    "rounding error. The agreement with BP is not robust; it is contingent.",
    "<b>Error propagates as a wavefront whose speed is set by graph topology.</b> "
    "The number of inference steps before a node first receives a non-zero error "
    "equals its <i>shortest path to the output</i>, exactly. For the chain I get "
    "arrival times [5,4,3,2,1,0]; adding explicit skips gives [3,2,2,1,1,0]. The "
    "consequence is a floor on the inference budget: <b>T &ge; L</b> for explicit "
    "skips versus <b>T &ge; 2L</b> for folded ones. Skip connections are not a "
    "detail in PC &mdash; they halve the required relaxation.",
])
para("So if PC is worth pursuing here, it is not for gradient quality. The "
     "question I actually care about is whether <b>the relaxation process "
     "produces different (and for patch prediction, better) latent "
     "representations</b> than a forward pass does. That is a question about "
     "representations, not about credit assignment, and as far as I can tell it "
     "is open.")

# ---------------------------------------------------------------- 2
para("2. Scaling the architecture: reproducing &micro;PC", h1)
para("To get to anything JEPA-shaped I need depth, residual connections, and "
     "autodiff. I therefore worked through &micro;PC (Innocenti et al.), which "
     "reparameterises PC networks in a depth-&micro;P style so that deep residual "
     "PCNs train at all. I reproduced their setting from scratch: a "
     "<b>30-layer fully-connected ResNet, width 128, on MNIST</b>, using the "
     "jpc library, with a standard-parameterisation (SP) control at identical "
     "depth, weights and learning rates.")

A(Image("fig_mupc_results.png", width=6.85 * inch, height=2.15 * inch))
para("<b>Figure 1.</b> All three panels are from my own runs at depth 30, "
     "width 128, batch 64, activity lr 0.5, weight lr 0.1 (Adam), seed 4329. "
     "(a) Mean activity norm per layer after a forward pass at initialisation; "
     "log scale; the output layer is omitted since its scaling differs by "
     "construction. (b) SP training loss per iteration until it becomes "
     "non-finite. (c) &micro;PC test accuracy over 900 iterations "
     "(&asymp;1 epoch), full 10k test set.", cap)

para("The control is unambiguous. Under SP the forward pass grows "
     "<b>8.5&times;</b> from the first to the last hidden layer at "
     "initialisation (6.53 &rarr; 55.55), and training diverges within six "
     "iterations &mdash; the loss goes 36.5, 1.0&times;10<sup>16</sup>, "
     "2.1&times;10<sup>23</sup>, 1.9&times;10<sup>28</sup>, "
     "1.7&times;10<sup>33</sup>, inf. Under &micro;PC the same forward pass grows "
     "<b>1.23&times;</b> (11.17 &rarr; 13.73) and the network trains smoothly to "
     "<b>93.61%</b> test accuracy in a single epoch. This is the paper&rsquo;s "
     "headline claim, reproduced independently.")

# ---------------------------------------------------------------- 3
para("3. The result that actually surprised me", h1)
para("I instrumented the training loop to record the PC energy <i>F(z)</i> at "
     "every one of the T = 30 inference steps, so I could check whether the "
     "relaxation was doing any work. It is doing almost none:")

A(tbl([["training iteration", "F(z) before inference", "F(z) after T=30 steps",
        "relative drop"],
       ["100", "0.1618", "0.1610", "0.49%"],
       ["200", "0.1228", "0.1222", "0.49%"],
       ["300", "0.0978", "0.0973", "0.51%"],
       ["400", "0.0935", "0.0930", "0.53%"]],
      [1.5 * inch, 1.65 * inch, 1.75 * inch, 1.0 * inch], align_right=(1, 2, 3)))
A(Spacer(1, 7))

para("Thirty steps of gradient descent on the activities move the energy by "
     "half a percent, and the figure is stable across training. In other words, "
     "under &micro;PC <b>the feedforward initialisation already lands essentially "
     "at the inference fixed point</b>. That is a coherent reading of why "
     "&micro;PC works &mdash; the paper shows the inference landscape becomes "
     "badly ill-conditioned with depth, and the reparameterisation appears to "
     "sidestep that by making relaxation nearly unnecessary rather than by making "
     "it converge faster.")
para("But it is awkward for my project. If inference is a 0.5% correction to a "
     "forward pass, then the latents PC produces are, to a very good "
     "approximation, the latents a forward pass produces &mdash; and the "
     "&ldquo;richer representations&rdquo; hypothesis from &sect;1 has little "
     "room to be true in this regime. Either the effect I am looking for lives "
     "somewhere other than the discriminative-MNIST setting, or it needs a "
     "network where relaxation genuinely moves the activities. <b>This is the "
     "single point I would most like a second opinion on.</b>")
para("I should flag a limitation honestly: the depth sweep across "
     "{8, 16, 30, 64} that would show how this scales is written but has not "
     "finished running, and my current numbers are single-seed. I am not "
     "claiming statistical strength for the 0.5% figure &mdash; only that it is "
     "consistent across four widely-separated logging points in one run.", small)

# ---------------------------------------------------------------- 4
para("4. A direction I have just been pointed to: MPC / NeuroSSL", h1)
para("Prof. Alexander Ororbia, whom I contacted about this project, pointed me "
     "to <b>Meta-Representational Predictive Coding</b> (Ororbia, Friston and "
     "Rao, arXiv:2503.21796), which he describes as the first encoder-centric "
     "or joint-embedding form of predictive coding. From its abstract, MPC "
     "avoids learning a generative model of raw pixels by predicting "
     "<i>representations</i> of the input across parallel streams, giving an "
     "encoder-only learning and inference scheme, with representational "
     "dynamics driven by active sensory glimpsing. He also noted that MPC ties "
     "the feedback weights to the forward ones &mdash; the usual PC "
     "simplification &mdash; and that there is no reason a structure like mine "
     "could not instead <b>learn the error weights</b>, as in neural generative "
     "coding (NGC).")
para("<b>I have not read this paper yet</b>, so I am not going to characterise "
     "its results here; it is the next thing I read, and I am flagging it "
     "because it clearly changes the shape of the project. Two implications I "
     "can already see:")
bullets([
    "It suggests the joint-embedding + PC combination is viable in a form much "
    "closer to what I proposed than anything in the &micro;PC line, which is "
    "purely discriminative.",
    "Learned (untied) error weights are a concrete axis I had not considered at "
    "all. Everything I have built so far assumes tied feedback, which is exactly "
    "the assumption that made my BP-equivalence check come out at cosine 1.0. "
    "Untying it would break that equivalence &mdash; which, given &sect;1, might "
    "be the point rather than a problem.",
])

# ---------------------------------------------------------------- 5
para("5. What I have running, and what is next", h1)
A(tbl([["component", "status"],
       ["PC over arbitrary computation graphs (JAX), 3 topologies",
        "done, verified against autodiff"],
       ["Error-wavefront / inference-budget analysis",
        "done; T \u2265 L vs T \u2265 2L result is exact"],
       ["\u00b5PC 30-layer ResNet on MNIST + SP control",
        "done, reproduces the paper"],
       ["Per-step inference-energy instrumentation",
        "done; gives the 0.5% result above"],
       ["Depth sweep {8, 16, 30, 64}, multi-seed",
        "written, not yet run"],
       ["Swap MNIST classification for patch prediction",
        "next"],
       ["Read MPC; evaluate untied / learned error weights",
        "next"]],
      [3.5 * inch, 3.35 * inch]))
A(Spacer(1, 8))
para("The immediate plan is to keep the &micro;PC-parameterised residual "
     "encoder but replace the supervised objective with a JEPA-style one &mdash; "
     "predict the representation of a masked patch from the representation of "
     "the context &mdash; and then re-run the same energy instrumentation. If "
     "relaxation still only moves the energy half a percent in that setting, "
     "the representational hypothesis is in trouble and I would rather find "
     "that out in the next two weeks than in six months.")

# ---------------------------------------------------------------- 6
para("6. Questions I would like your opinion on", h1)
para("<b>1. Is the &ldquo;richer latents&rdquo; framing the right question, or "
     "am I chasing something that the 0.5% energy result has already answered "
     "negatively?</b> If the forward pass lands at the fixed point, is there a "
     "regime &mdash; different objective, weaker skips, smaller inference "
     "learning rate, untied weights &mdash; where relaxation would genuinely "
     "reshape the representation, or is that simply not what PC buys you?", q)
para("<b>2. How would you actually measure &ldquo;richer&rdquo;?</b> This is my "
     "biggest methodological gap. Candidates I have considered: linear-probe "
     "accuracy on held-out tasks, effective rank of the representation matrix, "
     "resistance to representational collapse, downstream patch-prediction error. "
     "None feels decisive, and the &micro;PC paper explicitly does not study "
     "representation quality at all.", q)
para("<b>3. Are there directions in this space I should know about?</b> The MPC "
     "pointer above reoriented the project in a single email; I would rather "
     "discover the rest of that literature now than after building the wrong "
     "thing. In particular, is anyone applying NGC-style learned error weights "
     "in a joint-embedding setting, and is there work on PC for masked "
     "prediction specifically?", q)
para("And the blunt version: <b>are these efforts pointed in a sensible "
     "direction?</b> I would rather be redirected early. All code, notebooks and "
     "figures are available if any of the above is worth looking at directly.")

para("References: Millidge, Tschantz &amp; Buckley, <i>Predictive Coding "
     "Approximates Backprop along Arbitrary Computation Graphs</i> (2020). "
     "Innocenti et al., <i>&micro;PC: Scaling Predictive Coding to 100+ Layer "
     "Networks</i>. Ororbia, Friston &amp; Rao, <i>Meta-Representational "
     "Predictive Coding</i>, arXiv:2503.21796. Assran et al., <i>I-JEPA</i> "
     "(2023). Implementation uses the jpc library (Buckley lab).", small)

doc = SimpleDocTemplate("pc_progress_report.pdf", pagesize=LETTER,
                        leftMargin=0.72 * inch, rightMargin=0.72 * inch,
                        topMargin=0.66 * inch, bottomMargin=0.62 * inch,
                        title="PC Predictors for Joint-Embedding World Models",
                        author="Yash Nagraj")
doc.build(S)

import pypdfium2 as pdfium
print("PAGES", len(pdfium.PdfDocument("pc_progress_report.pdf")))
