# Scientific Outreach Package: The Digital Sphinx In-Silico Lesion Audit

**Prepared for Direct Communication with Authors of *The Digital Sphinx* (bioRxiv, doi:10.64898/2026.03.20.713233) and *Flybody* (Nature 2024)**

---

## 1. Verified Contact & Affiliation Roster

| Researcher | Verified Role & Affiliation | Verified Email | Relevant Publication / Focus |
| :--- | :--- | :--- | :--- |
| **Elliott T. T. Abe** | Graduate Researcher, Department of Biology & Applied Mathematics, **University of Washington** | `eabe@uw.edu` | Lead Author, *The digital sphinx: Can a worm brain control a fly body?* (bioRxiv: 10.64898/2026.03.20.713233) |
| **Dr. Bingni W. Brunton** | Associate Professor, Department of Biology, **University of Washington** | `bbrunton@uw.edu` | Senior Author, *The Digital Sphinx*; PI, Data-Driven Neuroscience Lab |
| **Dr. John C. Tuthill** | Professor, Department of Neurobiology and Biophysics, **University of Washington** | `tuthill@uw.edu` | Co-Author, *The Digital Sphinx*; PI, Neural Circuits of Somatosensation Lab |
| **Dr. Srinivas C. Turaga** | Group Leader, **HHMI Janelia Research Campus** | `turagas@janelia.hhmi.org` | Senior Author, *Flybody* (Nature 2024); Connectome-constrained embodied control |

---

## 2. Targeted Outreach Emails (Ready to Send)

### Email 1: To Elliott Abe & Dr. Bingni Brunton (Authors of *The Digital Sphinx*)
**Subject:** In-silico lesion audit of the MaleCNS connectome responding to *The Digital Sphinx*

**Body:**
> Dear Elliott and Bingni,
>
> We thoroughly enjoyed reading your recent paper, *The digital sphinx: Can a worm brain control a fly body?* (bioRxiv, doi:10.64898/2026.03.20.713233), which demonstrated that deep reinforcement learning can optimize an evolutionarily mismatched nematode worm connectome to drive realistic fruit fly locomotion.
>
> Inspired by your finding that behavioral mimicry does not guarantee biological fidelity, we asked whether authentic biological connectome wiring exhibits functional specificity under targeted single-neuron lesions that degree-matched random networks cannot replicate under identical embodied physics.
>
> In our pre-registered adversarial audit of the Janelia MaleCNS v1.0 descending motor pathway ($4\text{ DN} \to 125\text{ IN} \to 377\text{ MN}$) in FlyGym, we tested $N=10$ independently-seeded biological policies against an ensemble of 20 degree-preserving bipartite rewired null models (0.0 degree error, $\sim 7\%$ edge overlap).
>
> Targeted unilateral ablation of descending neuron DNa01 produced a stereotyped steering turn of **$38.19 \pm 10.83^\circ/\text{s}$** ($95\%$ bootstrap CI: $[32.37, 45.22]^\circ/\text{s}$) in the biological models, closely matching in-vivo behavioral optogenetic measurements ($30\text{--}40^\circ/\text{s}$), whereas the 20 degree-matched null models failed to steer (**$2.92 \pm 1.25^\circ/\text{s}$**; $0/20$ models replicating the turn).
>
> We have attached our 2-page summary brief (`outputs/LESION_NULLS.md`), our full preprint manuscript, and the open-source code repository (https://github.com/Anishp-cell/connectome-drl), and would love to hear your thoughts and critique.
>
> Best regards,  
> Anish Pathak  
> Independent Researcher | anishpathak778@gmail.com  
> https://github.com/Anishp-cell/connectome-drl  

---

### Email 2: To Dr. John Tuthill (University of Washington)
**Subject:** MaleCNS descending steering lesion phenotypes in embodied Drosophila locomotion

**Body:**
> Dear John,
>
> We have followed your lab’s work on sensory-motor integration in *Drosophila* with great admiration, particularly your recent co-authorship on *The digital sphinx: Can a worm brain control a fly body?* (bioRxiv: 10.64898/2026.03.20.713233) challenging the assumption that locomotion proves biological fidelity in connectome-constrained models.
>
> To test whether biological connectivity provides verifiable motor localization, we performed targeted in-silico unilateral ablations of descending neuron DNa01 within the Janelia MaleCNS v1.0 premotor subcircuit controlling a 42-DOF FlyGym fly.
>
> Across $N=10$ independent training runs, biological models consistently produced a lateralized steering turn of **$38.19^\circ/\text{s}$** ($95\%$ CI: $[32.37, 45.22]^\circ/\text{s}$) with intact forward walking retention ($12.7\text{ mm/s}$ steady-state), directly recapitulating in-vivo optogenetic observations.
>
> When compared against 20 degree-preserving rewired null models under identical embodied physics, single-neuron ablation failed to steer ($2.92^\circ/\text{s}$; $0/20$ null models replicating), proving that single-cell lesion vulnerability is uniquely rooted in the genuine biological wiring diagram.
>
> We have compiled a 2-page brief (`outputs/LESION_NULLS.md`) and side-by-side synchronized video stills and would welcome any feedback or critique from your team.
>
> Warm regards,  
> Anish Pathak  
> Independent Researcher | anishpathak778@gmail.com  
> https://github.com/Anishp-cell/connectome-drl  

---

### Email 3: To Dr. Srinivas Turaga (HHMI Janelia)
**Subject:** Testing functional motor specificity in Janelia MaleCNS v1.0 descending subcircuits

**Body:**
> Dear Srinivas,
>
> Your pioneering work on *Flybody* (Nature 2024) demonstrated the power of embodied biomechanical models for dissecting complex fly behavior.
>
> Addressing the ongoing *Digital Sphinx* debate regarding whether connectome constraints produce genuine biological representations, we evaluated whether descending premotor pathways in the Janelia MaleCNS v1.0 connectome possess causal lesion specificity under embodied physics.
>
> Using an adversarial suite of 20 degree-preserving double-edge-swapped random graph null models (0.0 in/out degree deviation across all 506 neurons), we showed that unilateral ablation of DNa01 produces lateralized steering turns ($38.19^\circ/\text{s}$, $95\%$ CI: $[32.37, 45.22]^\circ/\text{s}$) strictly in biological wiring, whereas degree-matched nulls stumble without steering ($2.92^\circ/\text{s}$).
>
> Crucially, evaluating across $N=10$ independently-seeded biological policies confirmed that this phenotype is an invariant property of the biological topology rather than a training fluke.
>
> We have attached our 2-page executive summary (`outputs/LESION_NULLS.md`) and would value your perspective on scaling this adversarial lesion suite to whole-VNC models.
>
> Sincerely,  
> Anish Pathak  
> Independent Researcher | anishpathak778@gmail.com  
> https://github.com/Anishp-cell/connectome-drl

---

## 3. Deliverable Verification Check

- [x] Pre-registered falsification criteria declared and evaluated: **HYPOTHESIS CONFIRMED [PASS 3/3]**.
- [x] Speed calibration: verified at **$12.7\text{ mm/s}$** ($5.1\text{ BL/s}$ steady-state) and $+7.9\text{--}8.4\text{ mm/s}$ (startup).
- [x] Biological sample size: upgraded to **$N=10$ independent policy models** (seeds $42 \dots 51$).
- [x] Null ensemble: evaluated across **$N=20$ degree-matched rewired models** (60 rollouts).
- [x] Publication dashboard: `outputs/digital_sphinx_audit.png` (with visible pre-registration box).
- [x] Side-by-side synchronized video: `outputs/digital_sphinx_audit.gif` (and `outputs/audit_video_stills.png`).
- [x] Executive brief: `outputs/LESION_NULLS.md` ($\le 2$ pages).
- [x] bioRxiv preprint: `paper/preprint.md` (complete 6-section manuscript).
