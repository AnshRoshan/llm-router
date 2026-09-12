# SOURCES.md — Complete Source Register with URLs and Dates

**Compiled 2026-09-12.** Every source consulted, with URL, date, and reliability tier.
Reliability: **[P]** = primary source read directly this session · **[S]** = single reputable secondary
source, primary not read · **[V]** = vendor/marketing · **[X]** = unverified or contradictory.

---

## A. Primary sources read directly this session [P]

### Repositories (fetched raw README / repo page)

| Source | URL | Date / status |
|---|---|---|
| lm-sys/RouteLLM (repo page: stars, forks, last commit, file listing) | https://github.com/lm-sys/RouteLLM | Last commit on `main` **2024-08-10**; 175 commits; 0 tags; ~5.5k★ / 430 forks |
| RouteLLM README (full, both chunks) | https://raw.githubusercontent.com/lm-sys/RouteLLM/main/README.md | README last updated 2024-07-20 |
| ulab-uiuc/LLMRouter README | https://github.com/ulab-uiuc/LLMRouter | Active; news entries through 2026-08 |
| `llmrouter-lib` PyPI JSON API | https://pypi.org/pypi/llmrouter-lib/json | **v0.4.0, uploaded 2026-08-14**; MIT; `Development Status :: 3 - Alpha`; Python 3.10–3.11; authors Tao Feng, Haozhen Zhang |
| `routellm` on PyPI | https://pypi.org/project/routellm/ | **v0.2.0, 2024-07-08** |
| Not-Diamond/awesome-ai-model-routing README | https://github.com/Not-Diamond/awesome-ai-model-routing | Curated list; papers through 2024-10 |
| MilkThink-Lab/Awesome-Routing-LLMs README (chunk 0) | https://github.com/MilkThink-Lab/Awesome-Routing-LLMs | Taxonomy by pre-judgment / verification routing; venues through 2026 |
| RouteWorks/RouterArena README + live leaderboard | https://github.com/RouteWorks/RouterArena | 26-router live leaderboard; RouteLLM **#25** |
| ZhangYiqun018/Avengers | https://github.com/ZhangYiqun018/Avengers | AAAI 2026 |
| ZhangYiqun018/AvengersPro | https://github.com/ZhangYiqun018/AvengersPro | Code for Avengers-Pro |
| Not-Diamond/RoRF | https://github.com/Not-Diamond/RoRF | ~200★ |
| aurelio-labs/semantic-router | https://github.com/aurelio-labs/semantic-router | ~3.7k★, MIT |
| withmartian/routerbench | https://github.com/withmartian/routerbench | Dataset + code |
| zhuchichi56/RouterXBench | https://github.com/zhuchichi56/RouterXBench | Code for ProbeDirichlet |
| somerstep/CARROT | https://github.com/somerstep/CARROT | Code for CARROT |
| Mercidaiha/IRT-Router | https://github.com/Mercidaiha/IRT-Router | Code for IRT-Router |
| yanweiyue/masrouter | https://github.com/yanweiyue/masrouter | ACL 2025 |
| epfl-dlab/forc | https://github.com/epfl-dlab/forc | FORC, WSDM 2024 |
| TZData1/llm-inference-router | https://github.com/TZData1/llm-inference-router | GreenServ |
| dongyuanjushi/OmniRouter | https://github.com/dongyuanjushi/OmniRouter | Agent-routing benchmark |
| knkumar/router_benchmark | https://github.com/knkumar/router_benchmark | Task/session-level routing harness |
| ypollak2/llm-router | https://github.com/ypollak2/llm-router | `pip install llm-routing` |
| ulab-uiuc/RouteProfile | https://github.com/ulab-uiuc/RouteProfile | arXiv:2605.00180 |
| HuggingFace: `ulab-ai/xRouteBench` | https://huggingface.co/datasets/ulab-ai/xRouteBench | LLMRouter benchmark dataset |
| HuggingFace: `lmsys/lmsys-arena-human-preference-55k` | https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k | RouteLLM's calibration dataset |
| HuggingFace: `routellm` org (router checkpoints) | https://huggingface.co/routellm | 2024-era weights |
| anyscale/llm-router (causal-LLM training notebook) | https://github.com/anyscale/llm-router | Referenced by RouteLLM README |

### Official documentation

| Source | URL | Date |
|---|---|---|
| LiteLLM — Router / Load Balancing | https://docs.litellm.ai/docs/routing | Read 2026-09-12; confirms `simple-shuffle` default + "Lowest Cost Routing (Async)" + Rate-Limit Aware v2 (ASYNC) |
| LiteLLM — proxy config settings (`router_settings` reference) | https://docs.litellm.ai/docs/proxy/config_settings | Full `router_settings` key list |
| LiteLLM — Routing Groups (per-model strategies) | https://docs.litellm.ai/docs/proxy/ui/routing_groups | — |
| LiteLLM — Router settings for keys and teams | https://docs.litellm.ai/docs/proxy/keys_teams_router_settings | Per-tenant routing overrides |
| LiteLLM — proxy reliability | https://docs.litellm.ai/docs/proxy/reliability | — |
| SGLang Model Gateway docs | https://docs.sglang.io/docs/advanced_features/sgl_model_gateway | 40+ Prometheus metrics, OTel, circuit breakers, token-bucket rate limiting |
| `sglang-router` on PyPI | https://pypi.org/project/sglang-router/ | 2026-01-15 |
| Aurelio Semantic Router — threshold optimization | https://docs.aurelio.ai/semantic-router/user-guide/features/threshold-optimization | `fit()`/`evaluate()`; 34.85% → 89.39% example |
| Aurelio Semantic Router — introduction | https://docs.aurelio.ai/semantic-router/get-started/introduction | — |
| vLLM Production Stack — semantic router integration | https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/semantic-router-integration.html | Envoy ExtProc + BERT/decoder-only LoRA |
| Not Diamond docs — routing between RAG agents | https://docs.notdiamond.ai/docs/routing-between-rag-agents | `LLMConfig`, `tradeoff: "cost"`, LangChain integration |
| Portkey — LiteLLM alternatives | https://portkey.ai/alternatives/litellm-alternatives | — |

### arXiv papers / proceedings read this session

| Paper | URL | Submitted / venue |
|---|---|---|
| **RouteLLM: Learning to Route LLMs with Preference Data** | https://arxiv.org/abs/2406.18665 · https://arxiv.org/html/2406.18665v4 | 2024-06; **ICLR 2025** |
| **Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey** (Moslem & Kelleher) | https://arxiv.org/abs/2603.04445 · https://arxiv.org/html/2603.04445v2 | v1 2026-02-23, **v2 2026-04-21** |
| **vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models** (32 authors) | https://arxiv.org/abs/2603.04444 · https://arxiv.org/html/2603.04444v1 | v1 2026-02-23, **v4 2026-06-03** |
| **When to Reason: Semantic Router for vLLM** (Wang et al., IBM/Tencent/UChicago) | https://arxiv.org/abs/2510.08731 · https://arxiv.org/html/2510.08731v1 | 2025-10-09; **NeurIPS 2025 ML-for-Systems Workshop** |
| **RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers** | https://arxiv.org/abs/2510.00202 · https://arxiv.org/html/2510.00202v1 · https://proceedings.iclr.cc/paper_files/paper/2026/file/4987bb24bc53c198785922d1bd9e18cf-Paper-Conference.pdf | 2025-09-30; **published at ICLR 2026** |
| **MixLLM: Dynamic Routing in Mixed Large Language Models** (Wang et al.) | https://arxiv.org/abs/2502.18482 · https://aclanthology.org/2025.naacl-long.545/ · https://aclanthology.org/2025.naacl-long.545.pdf | 2025-02-09; **NAACL 2025**, pp. 10912–10922 |
| **Adaptive LLM Routing under Budget Constraints (PILOT)** (Panda et al., Fujitsu) | https://arxiv.org/abs/2508.21141 · https://arxiv.org/html/2508.21141v1 · https://aclanthology.org/2025.findings-emnlp.1301/ | v1 2025-08-28, v2 2025-09-09; **EMNLP 2025 Findings** |
| **FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance** (Chen, Zaharia, Zou) | https://arxiv.org/abs/2305.05176 · https://arxiv.org/pdf/2305.05176 | 2023-05-09; **TMLR 2024** |
| **A Unified Approach to Routing and Cascading for LLMs** (Dekoninck, Baader, Vechev — ETH SRI) | https://arxiv.org/abs/2410.10347 · https://www.sri.inf.ethz.ch/publications/dekoninck2024cascaderouting | 2024-10; ICML 2025 |
| **CP-Router: An Uncertainty-Aware Router Between LLM and LRM** (Su et al.) | https://arxiv.org/abs/2505.19970 · https://arxiv.org/html/2505.19970v1 · https://ojs.aaai.org/index.php/AAAI/article/view/40589 | 2025-05; **AAAI 2026** |
| **UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing** | https://arxiv.org/html/2605.18796 | 2026 |
| **Conformal Cascade: Distribution-Free Accuracy Guarantees** | https://arxiv.org/pdf/2607.25018 | 2026-07 |
| **CARROT: A Cost Aware Rate Optimal Router** (Somerstep et al.) | https://arxiv.org/abs/2502.03261 | v1 2025-02-05, v2 2025-05-19; ICLR 2025 Workshop. Introduces the **SPROUT** dataset |
| **Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems** (RouterXBench / ProbeDirichlet, Wu et al.) | https://arxiv.org/abs/2602.11877 · https://arxiv.org/html/2602.11877v1 | 2026-02-12 |
| **LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing** | https://arxiv.org/html/2601.07206v1 | 2026-01 |
| **Beyond GPT-5: Making LLMs Cheaper and Better via Performance–Efficiency Optimized Routing (Avengers-Pro)** | https://arxiv.org/html/2508.12631 · https://dl.acm.org/doi/10.1145/3772429.3772445 | 2025; **CIKM '25** |
| **The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants** | https://arxiv.org/pdf/2505.19797 · https://ojs.aaai.org/index.php/AAAI/article/view/40790/44751 | 2025-05; **AAAI 2026** |
| **GraphRouter: A Graph-based Router for LLM Selections** (Feng, Shen, You — UIUC) | https://arxiv.org/html/2410.03834 · https://arxiv.org/pdf/2410.03834 · https://openreview.net/pdf?id=eU39PDsZtT | 2024-10; **ICLR 2025** |
| **PersonalizedRouter: Personalized LLM Routing via Graph-based User Preference Modeling** | https://arxiv.org/html/2511.16883v1 · https://arxiv.org/pdf/2511.16883 | **TMLR 11/2025** |
| **Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques** | https://arxiv.org/html/2506.06579v1 | 2025-06 |
| **Doing More with Less — Implementing Routing Strategies in LLM-Based Systems: An Extended Survey** | https://arxiv.org/html/2502.00409v2 | 2025-02 |
| **RouterBench: A Benchmark for Multi-LLM Routing System** (Hu et al.; originally submitted as "MARS") | https://arxiv.org/abs/2403.12031 · https://openreview.net/pdf?id=C0rs3wM0N8 · https://www.semanticscholar.org/paper/a2ca6f085007d0dceafbc09c2df24e70e771eac5 · https://withmartian.com/post/introducing-routerbench | 2024-03-16; Martian + UC Berkeley + UCSD |
| **RouterEval: A Comprehensive Benchmark for Routing LLMs** (Huang et al., Sun Yat-sen) | https://aclanthology.org/2025.findings-emnlp.208.pdf | **Findings of EMNLP 2025**, pp. 3860–3887, Nov 4–9 2025 |
| **Task- and Session-Level Model Routing: A Common-Interface Hybrid Evaluation of Four Open-Source Routers Across Four Benchmarks** | https://arxiv.org/html/2608.14641 | 2026-07-28 |
| **Dynamic LLM Routing and Selection based on User Preferences (OptiRoute)** | https://arxiv.org/pdf/2502.16696 | 2024-11 |
| **WISERouter: LLM Routing with Workload Budget Constraint** | https://arxiv.org/pdf/2607.23765 | 2026-07 |
| Closed-loop budget-pacing / hot-swap registry router (PILOT/PROTEUS gap analysis; 9.8 ms CPU routing) | https://arxiv.org/html/2604.00136v1 | 2026-03 |
| **Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving** | https://arxiv.org/html/2606.27457v1 | 2026-06 |
| **LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers** (Feng et al.) | https://arxiv.org/abs/2608.06867 · https://huggingface.co/papers/2608.06867 | 2026-08; HF Daily Papers |
| **TSRouter** (multimodal time-series routing) | https://arxiv.org/abs/2607.08940v1 | 2026-07 |
| **RouteProfile** | https://arxiv.org/abs/2605.00180 | 2026-05 |
| **Causal LLM Routing: End-to-End Regret Minimization from Observational Data** | https://arxiv.org/abs/2505.16037 | NeurIPS 2025 |
| **R2-Router: A New Paradigm for LLM Routing with Reasoning** | https://arxiv.org/abs/2602.02823 | ICML 2026 |
| **Uncertainty-Based Two-Tier Selection** | https://arxiv.org/abs/2405.02134 | COLM 2024 |
| **Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing** | https://arxiv.org/abs/2404.14618 | 2024-04 |
| **AutoMix: Automatically Mixing Language Models** | https://arxiv.org/abs/2310.12963 | NeurIPS 2024 |
| **MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs** | https://arxiv.org/abs/2407.10834 · https://github.com/mail-research/MetaLLM-wrapper/ | 2024-07 |
| **OptLLM: Optimal Assignment of Queries to Large Language Models** | https://arxiv.org/abs/2405.15130 | 2024-05 |
| **Routoo: Learning to Route to Large Language Models Effectively** | https://arxiv.org/abs/2401.13979 | 2024-01 |
| **Fly-Swat or Cannon? Cost-Effective LM Choice via Meta-Modeling (FORC)** | https://arxiv.org/abs/2308.06077 | WSDM 2024 |
| **Routing to the Expert: Efficient Reward-guided Ensemble of LLMs (Zooter)** | https://arxiv.org/abs/2311.08692 | NAACL 2024 |
| **Cache & Distil: Optimising API Calls to Large Language Models** | https://arxiv.org/abs/2310.13561 | ACL 2024 |
| **EcoAssistant: Using LLM Assistant More Affordably and Accurately** | https://arxiv.org/abs/2310.03046 | 2023-10 |
| **Large Language Model Routing with Benchmark Datasets** | https://arxiv.org/abs/2309.15789 | 2023-09 |
| **Tryage: Real-time, intelligent Routing of User Prompts to LLMs** | https://arxiv.org/abs/2308.11601 | 2023-08 |
| **LLM-Blender** | https://arxiv.org/abs/2306.02561 | 2023-06 |
| **EmbedLLM: Learning Compact Representations of Large Language Models** | https://arxiv.org/abs/2410.02223 | 2024-10 |
| **Harnessing the Power of Multiple Minds: Lessons Learned from LLM Routing** | https://arxiv.org/abs/2405.00467 | 2024-05 |
| **Merge, Ensemble, and Cooperate! (survey)** | https://arxiv.org/abs/2407.06089 | 2024-07 |
| **Towards Generalized Routing: Model and Agent Orchestration** | https://arxiv.org/abs/2509.07571 | 2025 |
| **RADAR: Reasoning-Ability and Difficulty-Aware Routing** | https://arxiv.org/abs/2509.25426 | NeurIPS 2025 Workshop |
| **LLMRank: Understanding LLM Strengths for Model Routing** | https://arxiv.org/abs/2510.01234 | 2025 |
| **PROTEUS: SLA-Aware Routing via Lagrangian RL** | https://arxiv.org/abs/2601.19402 | 2026 |
| **IPR: Intelligent Prompt Routing with User-Controlled Quality-Cost Trade-offs** | https://aclanthology.org/2025.emnlp-industry.170/ | EMNLP 2025 Industry |
| **IRT-Router** | https://aclanthology.org/2025.acl-long.761/ | ACL 2025 |
| **MasRouter: Learning to Route LLMs for Multi-Agent Systems** | https://arxiv.org/abs/2502.11133 | ACL 2025 |
| **OmniRouter: Budget and Performance Controllable Multi-LLM Routing** | https://arxiv.org/abs/2502.20576 | KDD 2025 |
| **Near-Optimal Online Deployment and Routing for Streaming LLMs** | https://arxiv.org/abs/2506.17254 | ICLR 2026 |
| **Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing** | https://arxiv.org/abs/2507.15553 | 2025 |
| **GreenServ: Energy-Efficient Context-Aware Dynamic Routing** | https://arxiv.org/abs/2601.17551 | 2026 |
| **Federate the Router** | https://arxiv.org/abs/2601.22318 | 2026 |
| **Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization** | https://arxiv.org/abs/2603.12933 | 2026 |
| **Truthful Reverse Auctions for Adaptive Selection via Contextual Multi-Armed Bandits** | https://arxiv.org/abs/2602.14476 | AAMAS 2026 |
| **Cost-Aware Routing for Efficient Text-To-Image Generation** | https://arxiv.org/abs/2506.14753 | 2025 |
| **Dynamic Quality-Latency Aware Routing in Wireless Edge-Device Networks** | https://ieeexplore.ieee.org/abstract/document/11147210 | ICCC Workshop 2025 |
| **GraphRAG-Router** | https://pith.science/paper/2604.16401 | 2026 |
| **Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale** | https://arxiv.org/html/2601.01330 | 2026-02 |
| **GraphPlanner: Graph Memory-Augmented Routing** | https://openreview.net/pdf?id=ZdGB7MNQDT | — |

---

## B. Secondary / practitioner sources [S]

| Source | URL | Date |
|---|---|---|
| RouteLLM vs NotDiamond vs Martian: Do LLM Model Routers Actually Cut Costs? | https://dreaming.press/posts/2026-06-21-routellm-vs-notdiamond-vs-martian.html | 2026-06-22 |
| LLM Gateways Compared 2026: LiteLLM vs OpenRouter vs Portkey vs RouteLLM | https://wavect.io/blog/llm-gateway-router-comparison-2026/ | 2026-07-07 (pub 2026-06-27) |
| LLM Routers Compared: LiteLLM vs Portkey vs OpenRouter in 2026 | https://www.developersdigest.tech/blog/llm-router-comparison-2026 | 2026-06-24 |
| The LLM Gateway & Router Index (2026) | https://aiprosol.com/llm-gateways | 2026-06-13 |
| Best LLM Gateways and API Routers in 2026 | https://devtoollab.com/blog/best-llm-gateways | 2026-07-28 |
| 8 Best LLM Routers in 2026 (DigitalOcean) | https://www.digitalocean.com/resources/articles/best-llm-routers | 2026-08-18 |
| Best LLM Router and AI Gateway (2026) — Inworld | https://inworld.ai/resources/best-llm-router-ai-gateway | 2026-03-26 |
| Portkey vs LiteLLM vs OpenRouter — PkgPulse (adoption/star counts) | https://www.pkgpulse.com/guides/portkey-vs-litellm-vs-openrouter-llm-gateway-2026 | 2026-03-09 |
| OpenRouter vs LiteLLM vs Portkey 2026 | https://toolhalla.ai/blog/openrouter-vs-litellm-vs-portkey-2026 | 2026-03-21 |
| Model Router Buyer's Guide: Buy Failover, Not Judgment | https://www.beri.net/article/model-router-buyers-guide-failover-not-judgment-2026 | 2026-09-12 |
| RouteLLM in Production: Dynamic Cascades (Effloow) | https://effloow.com/articles/routellm-hybrid-model-routing-cost-optimization-poc-2026 | 2026-08-21 |
| LiteLLM Semantic Routing: RouteLLM vs vLLM Semantic Router | https://gingerlabs.ai/blog/routellm-vs-vllm-semantic-router | 2026-06-20 |
| LiteLLM Router Setup: Fallback, Cost Routing & Model Pools | https://www.gingerlabs.ai/blog/litellm-router-setup-guide | 2026-06-13 |
| LiteLLM Fallback Configuration | https://markaicode.com/tutorial/litellm-fallback-configuration/ | 2026-05-25 |
| LiteLLM AI Gateway: Route Local + Cloud Models | https://localaimaster.com/blog/ai-gateway-litellm | 2026-04-23 |
| LiteLLM Architecture: Production LLM Gateway on Kubernetes | https://markaicode.com/architecture/litellm-architecture/ | 2026-07-26 |
| Circuit Breaker Architecture for LLM API Resilience | https://markaicode.com/architecture/circuit-breaker-resilient-ai-systems/ | 2026-07-24 |
| LangChain Inference Architecture: Production System Design | https://markaicode.com/architecture/langchain-inference-architecture/ | 2026-07-26 |
| LangChain Fault-Tolerant Architecture for Production AI | https://markaicode.com/architecture/tool-fault-tolerant-architecture/ | 2026-07-30 |
| Model Routing and Cascades for LLM Cost (TMLS) | https://www.tmls.nyc/research/model-routing-cascades | 2026-06-02 |
| LLM Model Routing: Route Queries Automatically (NeuralTrust) | https://neuraltrust.ai/blog/llm-model-routing | 2026-07-22 |
| LLM Model Routing in 2026: Cost-Quality Optimization | https://www.digitalapplied.com/blog/llm-model-routing-2026-cost-quality-optimization-engineering-guide | 2026-06-14 |
| How LLM Routing Cuts Agent Costs 40–85% in Production | https://www.channel.tel/blog/llm-routing-cx-agents-cost-quality | 2026-07-04 |
| LLM Routing Can Cost More Than Not Routing | https://blog.dailydoseofds.com/p/llm-routing-can-cost-more-than-not | 2026-09-07 |
| LLM Model Routing: How to Cut Cost Without Losing Quality | https://agentropic.ai/blog/llm-model-routing-cost/ | 2026 |
| Model Routing: Send Easy Queries to Cheap Models | https://ai-tldr.dev/learn/production-llmops/llmops-fundamentals/llm-model-routing/ | 2026-06-12 |
| Agent Router in ML Systems — Complete Guide (2026) | https://123ofai.com/articles/blocks/agent-router | 2026-02-07 |
| Semantic Router: Embedding-Based Routing Without Calling an LLM | https://sureprompts.com/blog/semantic-router-implementation | 2026-04-22 |
| AI Agent Model Routing and Dynamic Model Selection Strategies | https://zylos.ai/research/2026-03-02-ai-agent-model-routing/ | 2026-03-02 |
| LLM Routing: The Secret Architecture Cutting AI Costs by 60% | https://app.ailog.fr/en/blog/guides/llm-routing-cost-optimization | 2026-08-03 |
| Three-Tier LLM Routing / How to Set Up an AI Model Router | https://www.mindstudio.ai/blog/set-up-ai-model-router-llm-stack-c2610 · https://www.mindstudio.ai/blog/set-up-ai-model-router-llm-stack | 2026-02-10 |
| LLM routing in production: Choosing the right model for every request | https://blog.logrocket.com/llm-routing-right-model-for-requests/ | 2026-03-27 |
| Best LLM Observability Tools in 2026 | https://www.firecrawl.dev/blog/best-llm-observability-tools | 2025-12-02 |
| 4 best LLM gateways for observability (Braintrust) | https://www.braintrust.dev/articles/best-llm-gateways-observability-2026 | 2026-03-27 |
| Top 5 LLM Gateways in 2026 (Maxim) — gateway latency/throughput comparison | https://www.getmaxim.ai/articles/top-5-llm-gateways-in-2025-the-definitive-guide-for-production-ai-applications/ | 2026-05-27 |
| FrugalGPT cascade routing implementation tutorial (Nadir) | https://getnadir.com/blog/frugalgpt-cascade-routing-implementation-tutorial/ | 2026-06-29 |
| RouterArena paper note (ICLR 2026) | https://en.papernotes.org/ICLR2026/llm_evaluation/routerarena_an_open_platform_for_comprehensive_comparison_of_llm_routers/ | 2026-05-08 |
| RouterArena topic overview (EmergentMind) | https://www.emergentmind.com/topics/routerarena | — |
| RouterBench Dataset topic overview (EmergentMind) | https://www.emergentmind.com/topics/routerbench-dataset | 2025-09-02 |
| Avengers-Pro framework topic overview (EmergentMind) | https://www.emergentmind.com/topics/avengers-pro-framework | 2025-08-24 |
| vLLM semantic router explainer | https://www.arunbaby.com/ml-system-design/0076-vllm-semantic-router-multi-model-inference/ | 2026-02-14 |
| MixLLM summary (Lacuna) | https://lacuna.tiptreesystems.com/work/mixllm-dynamic-routing-in-mixed-large-language-models/wrk_415944a6253a6b3a1521be12390cfae9 | 2025-09-12 |
| MixLLM blog (NEC Labs) | https://www.nec-labs.com/blog/mixllm-dynamic-routing-in-mixed-large-language-models/ | 2025-08-27 |
| MixLLM quick review (Liner) | https://liner.com/review/mixllm-dynamic-routing-in-mixed-large-language-models | 2025 |
| RouteLLM Manual (Doramagic project pack) | https://doramagic.ai/en/projects/routellm/manual/ | 2026-07-24 |
| RouteLLM deepwiki overview | https://deepwiki.com/lm-sys/RouteLLM | 2025-04-18 |
| semantic-router deepwiki overview | https://deepwiki.com/aurelio-labs/semantic-router | 2026-07-27 |
| Semantic Router glossary (Deepchecks) | https://www.deepchecks.com/glossary/semantic-router/ | 2025-03-03 |
| Semantic Router tool profile | https://ai-tldr.dev/tools/semantic-router-aurelio-labs/ | — |
| LLMRouter GraphRouter module README | https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/graphrouter/README.md | — |
| Using RouteLLM to Optimize LLM Usage (DevStackTips) | https://devstacktips.com/development/machine-learning/2025-08-10/using-routellm-to-optimize-llm-usage/ | 2025-08-10 |
| Performance Routing Optimization (Stanford CS224N final report — RouterBench critique) | https://web.stanford.edu/class/cs224n/final-reports/256847361.pdf | — |
| RoRF HN launch thread (Tomás, Not Diamond founder) | https://news.ycombinator.com/item?id=41636904 | 2024 |
| LLM Model Routing Compared: RouteLLM vs OpenRouter vs Not Diamond | https://nomadx.ae/blog/llm-model-routing-routellm-openrouter-notdiamond-2026/ | 2026-08-06 |
| OpenRouter Alternatives for Production | https://markaicode.com/alternatives/openrouter-alternatives/ | 2026-08-05 |
| AWS Bedrock pricing breakdowns | https://cloudburn.io/blog/amazon-bedrock-pricing · https://avahi.ai/blog/aws-bedrock-pricing/ · https://www.swfte.com/blog/aws-bedrock-guide-2026 | 2026-05-14 / 2026-06-29 / 2026-07-28 |
| OpenRouter vs AWS Bedrock (TrueFoundry) | https://www.truefoundry.com/blog/openrouter-vs-aws-bedrock | 2026-04-02 |
| Azure AI Foundry vs AWS Bedrock | https://myengineeringpath.dev/tools/azure-vs-bedrock/ | 2026-03-20 |
| Cost-based routing in production | https://theneuralbase.com/ai-in-production/learn/intermediate/cost-based-routing-in-production/ | 2026-04-11 |
| LLM Routing: Optimizing Pathways in Language Processing | https://medium.com/accredian/llm-routing-optimizing-pathways-in-language-processing-c52c2adf7c4e | 2024-10-04 |
| RouterBench explainer (Medium) | https://medium.com/@doubletaken/routerbench-a-game-changer-in-multi-llm-routing-systems-8b1452b5412f | 2025-02-06 |
| RouterBench announcement (MarkTechPost) | https://www.marktechpost.com/2024-03-30/routerbench-a-novel-machine-learning-framework-designed-to-systematically-assess-the-efficacy-of-llm-routing-systems/ | 2024-03-30 |
| FrugalGPT explainer (BayJarvis) | https://blog.bayjarvis.com/paper/frugalgpt-making-large-language-models-affordable-and-cheaper | 2023 |
| LiteLLM config.yaml structure reference (gist) | https://gist.github.com/yigitkonur/4053c1e31f9351da1d91a0ca410442fd | 2025-03-22 |
| Context7 RouteLLM index | https://context7.com/lm-sys/routellm | — |
| OpenTelemetry for AI Agents (MintMCP) | https://www.mintmcp.com/blog/opentelemetry-ai-agents | 2026-04-16 |

---

## C. Vendor / marketing sources [V] — use with caution

| Source | URL | Date | Claim |
|---|---|---|---|
| Microsoft Foundry Model Router hands-on demo | https://techcommunity.microsoft.com/blog/azuredevcommunityblog/optimising-ai-costs-with-microsoft-foundry-model-router/4494776 | pub 2026-02-27, mod 2026-02-13 | **4.5% / 4.7% / 14.2%** savings across Balanced/Cost/Quality modes on 10 prompts |
| Morph Router — what is an LLM router | https://www.morphllm.com/llm-router | 2026-03-31 | ~430 ms classification, $0.001/classification, 40–70% savings; classifier-method latency/accuracy table |
| Morph Router vs Not Diamond | https://www.morphllm.com/notdiamond-alternative | 2026-06-15 | Not Diamond trains across 60+ models; pricing model comparison |
| Not Diamond — RoRF blog | https://www.notdiamond.ai/blog/rorf-routing-on-random-forests-2 | 2024-09-25 | 12.5% cost reduction vs Claude 3.5 Sonnet; "outperforms RouteLLM" |
| Amazon Bedrock Intelligent Prompt Routing (dev.to AWS Builders) | https://dev.to/aws-builders/amazon-bedrock-intelligent-prompt-routing-cut-ai-costs-by-94-4m1k | 2026-03-27 | "94%" — this is a **per-token price comparison, not a measured routing saving**. Do not cite as a routing result |
| Amazon Bedrock Intelligent Prompt Routing (Medium/AWS Tip) | https://medium.com/aws-tip/amazon-bedrock-intelligent-prompt-routing-optimize-cost-and-quality-with-automatic-model-selection-4efa40bd8796 | 2025 | 30–56% savings; **~85 ms P90 overhead** |
| Bedrock IPR hands-on (Reddit r/aws) | https://www.reddit.com/r/aws/comments/1ttx97x/handson_amazon_bedrock_intelligent_prompt_routing/ | 2026-06 | ~65% savings at ~70/30 split; caveats: English-only, exactly 2 models, same family, not tunable |
| Anyscale — building an LLM router (RouteLLM collaborator) | https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses | 2024 | RouteLLM's commercial-comparison partner |
| LMSYS RouteLLM blog post | http://lmsys.org/blog/2024-07-01-routellm/ | 2024-07-01 | Origin of the 85% headline |
| FrugalGPT on r/singularity (original thread) | https://www.reddit.com/r/singularity/comments/13dnfd7/ | 2023-05-10 | Community reaction to the 98% claim |

---

## D. Unverified or contradictory — do not cite without further checking [X]

| Claim | Conflict | Where it appears |
|---|---|---|
| **Bedrock Intelligent Prompt Routing pricing** | `$1.00 per 1,000 routed requests` (cloudburn.io; DigitalOcean comparison table) **vs** "No separate fee per routing call; pay per underlying model tokens" (llmreference.com, https://www.llmreference.com/router/bedrock-intelligent-prompt-routing) | `02-frameworks-landscape.md` §2.3 — flagged |
| **Bedrock IPR savings %** | 30% (AWS docs, most-cited) vs 35%/56%/16% by family (beri.net) vs 60% vs Sonnet 3.5 v2 (llmreference.com "internal AWS test") vs 65% (Reddit anecdote) | All within-family. Flagged |
| **BEST-Route** — arXiv:2506.22716, Microsoft, ICML 2025, "15–30% fewer escalations" | Only ever seen via getnadir.com tutorial. arXiv page not read. | `03-technique-taxonomy.md` §3.4 — flagged Medium |
| **vLLM-SR "98.53% routing accuracy with 805 examples / 2h compute"** | Appears only in gingerlabs.ai blog, not in either arXiv paper I read | `02-frameworks-landscape.md` §2.4 — flagged Medium |
| **Martian "up to 97%, often beating GPT-4"** | No primary documentation found anywhere; closed product | `02-frameworks-landscape.md` — flagged |
| **"Routing typically cuts costs 60–75%"** | Repeated across marketing content (nomadx.ae, channel.tel) with no neutral basis found | Not adopted; contradicted by RouterArena (~35%) and UCCI (31%) |
| **Agentic routing works** | OmniRouter claims +6.30% accuracy / −10.15% cost; arXiv:2608.14641 found effects "inconclusive" on BFCL v4 / RouterBench / tau2-bench and "equivalent" on WebArena (TOST) | Both 2025–26. **Unresolved — measure it yourself** |
| **LLMRouterBench "most routing methods collapse to similar performance"** | Verified the paper exists (arXiv:2601.07206) and the phrasing is its own, but no independent replication found | `01-routellm-deep-dive.md` §6 |
| **`local-llm-router`, `ai-llm-router`** | PyPI pages only; repos not audited | `06-design-recommendations.md` §6.1 |

---

- **Anthropic 1-hour cache-write multiplier — 2.0× vs 1.5×.** Four sources say 2.0× (intuitionlabs 2026-09-05, technspire 2026-09-06, ofox 2026-06-10, aioutlooks); blog.sandbase.ai (2026-08-01) says 1.5× and computes with it. Use 2.0×, verify against the live pricing page. See §G.

## E. arXiv IDs cited across this dossier (70)

2305.05176 · 2306.02561 · 2308.06077 · 2308.11601 · 2309.15789 · 2310.03046 · 2310.12963 · 2310.13561 · 2311.08692 · 2401.13979 · 2403.12031 · 2404.14618 · 2405.00467 · 2405.02134 · 2405.15130 · 2406.18665 · 2407.06089 · 2407.10834 · 2410.02223 · 2410.03834 · 2410.10347 · 2502.00409 · 2502.03261 · 2502.11133 · 2502.16696 · 2502.18482 · 2502.20576 · 2505.16037 · 2505.19797 · 2505.19970 · 2505.23052 · 2506.03989 · 2506.06579 · 2506.14753 · 2506.17254 · 2506.22716 · 2507.15553 · 2508.12631 · 2508.21141 · 2509.07571 · 2509.25426 · 2510.00202 · 2510.01234 · 2510.08731 · 2511.16883 · 2512.09487 · 2601.01330 · 2601.07206 · 2601.17551 · 2601.19402 · 2601.22318 · 2602.02823 · 2602.11877 · 2602.14476 · 2603.04444 · 2603.04445 · 2603.12933 · 2604.00136 · 2604.03455 · 2604.16401 · 2605.00180 · 2605.18796 · 2606.02581 · 2606.27457 · 2607.08940 · 2607.23765 · 2607.25018 · 2608.06867 · 2608.14641 · 2608.19677


## F. Sources added in the revised-scope pass (workload × deployment classes)

### Primary [P]
| Source | URL | Date |
|---|---|---|
| **RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented LMs** | https://arxiv.org/abs/2505.23052 · https://arxiv.org/html/2505.23052v2 | v2 2025-10-17 |
| **Lightweight Query Routing for Adaptive RAG (RAGRouter-Bench study)** | https://arxiv.org/abs/2604.03455 · https://arxiv.org/html/2604.03455v1 | 2026-04 |
| **Cost-Aware Query Routing in RAG (CA-RAG)** | https://arxiv.org/abs/2606.02581 · https://arxiv.org/html/2606.02581v1 | 2026-03 |
| **RouteRAG: Efficient RAG from Text and Graph via RL** | https://arxiv.org/abs/2512.09487 | 2025-12 |
| **Stronger Baselines for RAG with Long-Context LMs (DOS RAG)** | https://arxiv.org/html/2506.03989 | 2026-01 |
| **Efficient RAG via Token Co-occurrence Graphs (TIGRAG)** | https://arxiv.org/html/2606.30093v1 | 2026 |
| **CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving** | https://arxiv.org/html/2608.19677 | 2026-08 |
| **Agentic Routing: The Harness-Native Data Flywheel** | https://arxiv.org/html/2607.11399v1 | 2026-07 |
| **Agent-as-a-Router: Agentic Model Routing for Coding Tasks** | https://arxiv.org/html/2606.22902v1 | 2026-06 |
| **Task- and Session-Level Model Routing** (deeper read: §6.9 utility, granularity table) | https://arxiv.org/html/2608.14641 | 2026-07-28 |
| **TSRouter** | https://arxiv.org/abs/2607.08940v1 | 2026-07 |
| Hybrid Cloud-Local LLM architecture guide (working `routeRequest` code, thresholds) | https://www.sitepoint.com/hybrid-cloudlocal-llm-the-complete-architecture-guide-2026/ | 2026-04-22 |
| OpenClaw issue #6421 — two-tier model routing config schema | https://github.com/openclaw/openclaw/issues/6421 | 2026-02-01 |

### Secondary [S]
| Source | URL | Date |
|---|---|---|
| **Claude Code Costs, Act II — model-scoped prompt cache, measured** | https://dev.to/sumedhbala/claude-code-costs-act-ii-where-the-big-hidden-costs-are-4gf1 | 2026-06-27 |
| Why Your Agent's Real Cost Is Its KV-Cache Hit Rate | https://dreaming.press/posts/kv-cache-hit-rate-the-metric-that-decides-your-agents-bill.html | 2026-07-28 |
| Context Engineering for Production AI Agents: KV Cache, Prefix Caching | https://www.spheron.network/blog/context-engineering-production-ai-agents-kv-cache-long-context/ | 2026-06-17 |
| **Cache-Aware Model Routing: Lower LLM Costs, Keep Quality** (cache-hit pricing table) | https://www.gmicloud.ai/en/blog/how-cache-aware-model-routing-reduces-llm-infrastructure-costs-without-sacrificing-quality | 2026-08-20 |
| KV Cache Optimization for LLM Inference Guide (PagedAttention vs RadixAttention) | https://www.gmicloud.ai/en/blog/kv-cache-optimization-for-llm-inference-how-cache-aware-serving-reduces-cost-and-latency | 2026-08-10 |
| **KV Cache Locality: The Hidden Variable in Your LLM Serving Cost** (measured 12.5% → 97.5%) | https://ranvier.systems/2026-04-30/kv-cache-locality-the-hidden-variable-in-your-llm-serving-cost.html | 2026-04-30 |
| GKE Inference Gateway: KV-Cache-Aware LLM Routing | https://www.spheron.network/blog/gke-inference-gateway-kv-cache-aware-llm-routing/ | 2026-06-22 |
| vLLM Prefix Caching Explained 2026 | https://packet.ai/blog/vllm-prefix-caching | 2026-08-05 |
| Keeping KV cache across turns on Apple Silicon (200× at 100K context) | https://www.reddit.com/r/LocalLLaMA/comments/1ru6hgp/ | 2026-03-15 |
| vLLM Agent Architecture: Production Design at Scale (sticky routing + prefix caching) | https://markaicode.com/architecture/vllm-agent-architecture/ | 2026-07-30 |
| A Comprehensive Guide to Model Routing (task/step/session levels) | https://www.notdiamond.ai/blog/a-comprehensive-guide-to-model-routing | 2026-04-23 |
| Hermes Multi-LLM Workflows | https://techjacksolutions.com/ai-tools/hermes/hermes-multi-llm-workflow/ | 2026-09-07 |
| Model Routing vs Tool Routing (WorkOS) | https://workos.com/blog/model-routing-vs-tool-routing-ai-agents | 2026-03-18 |
| Best AI Model for Coding Agents: A Routing Guide (per-role cost arithmetic) | https://www.augmentcode.com/guides/ai-model-routing-guide | 2026-06-18 |
| 9 Best LLM Routers and Model Routing Tools in 2026 | https://entelligence.ai/blogs/9-best-llm-routers-and-model-routing-tools-in-2026 | 2026-08-03 |
| Best LLMs for Building AI Agents in 2026 (tool-call reliability) | https://docs.promptise.com/blog/best-llm-for-ai-agents/ | 2026-07-16 |
| Why Multi-turn Agents Need More Than a Task Graph | https://blog.dailydoseofds.com/p/why-multi-turn-agents-need-more-than | 2026-09-10 |
| vLLM vs Anthropic API: When Self-Hosting Actually Pays Off | https://markaicode.com/vs/vllm-vs-anthropic-api/ | 2026-08-07 |
| vLLM vs OpenAI API | https://markaicode.com/vs/vllm-vs-openai-api/ | 2026-08-07 |
| vLLM vs Replicate | https://markaicode.com/vs/vllm-vs-replicate/ | 2026-08-07 |
| vLLM vs RunPod | https://markaicode.com/vs/vllm-vs-runpod/ | 2026-08-07 |
| Groq vs vLLM: LPU Speed vs GPU Control | https://markaicode.com/vs/groq-vs-vllm/ | 2026-07-22 |
| Self-Host LLMs or Use the API? Real $/1M Token Numbers in 2026 (5–15 min pod readiness) | https://tensoria.fr/en/blog/deploying-llms-to-production | 2026-05-15 |
| Self-Hosting Open-Weight LLMs: 2026 Decision Guide | https://www.digitalapplied.com/blog/self-hosting-open-weight-llms-2026-deployment-decision-guide | 2026-05-27 |
| **When to run a self-hosted LLM (Vercel)** — "settle routing before you buy GPUs" | https://vercel.com/i/self-hosted-llm | 2026-08-31 |
| Hybrid strategy: local + cloud overflow (the 100× cascade trap, working code) | https://theneuralbase.com/ollama/learn/advanced/hybrid-strategy-local-cloud-overflow/ | 2026-04-11 |
| The Hybrid AI Architecture: Route Local + Cloud | https://localaimaster.com/blog/hybrid-local-cloud-ai | 2026-04-11 |
| Local-First AI Agents: Hybrid Cloud-Edge Architectures | https://zylos.ai/research/2026-05-10-local-first-ai-agents-hybrid-cloud-edge-architectures/ | 2026-05-10 |
| Hybrid AI Architecture, Part 2: Routing Models to Reduce Cost | https://princetonits.com/hybrid-ai-architecture-part-2-routing-models-to-reduce-cost-without-reducing-quality/ | 2026-07-02 |
| Hybrid AI Architecture: Cloud Routing + Local Models | https://www.elinkdesign.com/hybrid-ai-architecture-cloud-routing-local-models-for-privacy-and-savings | 2026-04-14 |
| Hybrid LLM Architecture: API Models with Open-Source Runtimes | https://brics-econ.org/hybrid-llm-architecture-combining-api-models-with-open-source-runtimes | 2026-08-29 |
| vLLM self hosting vs API cost comparison | https://theneuralbase.com/vllm/qna/vllm-self-hosting-vs-api-cost-comparison/ | 2026-04-11 |

---

---

## G. Sources added for the session-stickiness pass (Thread 8), verified 2026-09-12

Cache pricing was **re-verified from scratch** this pass (not reused from §F snippets) because the
switch-cost algebra in `08` §8.3 depends on the write multiplier. Five independent sources agree on
Anthropic's `1.25× / 2.0× / 0.1×` and on OpenAI's **no-write-premium** structure.

| Source | URL | Date | Used for |
|---|---|---|---|
| LLM Prompt Caching: Cost Savings, Invalidation & Workload Design (cross-provider write/read/TTL/min-tokens table incl. GPT-5.6+ manual breakpoints) | https://intuitionlabs.ai/articles/llm-prompt-caching-cost-savings | 2026-09-05 | §8.1 table |
| Anthropic Claude prompt caching pricing: write, read, TTL math | https://technspire.com/en/blog/anthropic-prompt-caching-pricing-mechanics | 2026-09-06 | 1.25×/2×/0.1×; per-model minimums |
| Anthropic vs OpenAI Prompt Caching 2026: Cost Math + 3 Cache-Miss Fixes (**`prompt_cache_key`**; GPT-5.4/5.5 at 90%, not 50%; 3 canonical cache-miss patterns) | https://ofox.ai/blog/prompt-caching-cost-math-anthropic-vs-openai-2026/ | 2026-06-10 | §8.1, §8.4 Path A, §8.3(3) |
| Anthropic Prompt Caching in Production (cache key = byte-exact prefix + model + account; 4 breakpoints; gateway transparency; `cache_read_input_tokens` counters) | https://buzzai.cc/blog/anthropic-prompt-caching-playbook | 2026-05-24 | §8.1, §8.7(1) |
| Anthropic prompt caching, explained: two-tier write premium (per-N amortization table) | https://dev.to/rikuq/anthropic-prompt-caching-explained-cachecontrol-markers-the-two-tier-write-premium-and-when-it-25cp | 2026-06-14 | §8.3 amortization shape |
| OpenAI prompt caching, explained: automatic, free to enable (GPT-5.4 $2.50→$0.25/M; **no write premium**; "50% vs 90% framing is outdated") | https://dev.to/rikuq/openai-prompt-caching-explained-automatic-free-to-enable-90-off-cached-input-tokens-7bn | 2026-06-10 | §8.1 correction |
| OpenAI vs Anthropic Prompt Caching: Key Differences (hit-rate guarantee: OpenAI ~50% best-effort vs Anthropic ~100% when configured) | https://gingerlabs.ai/blog/openai-vs-anthropic-prompt-caching | 2026-06-17 | §8.7(1) |
| Anthropic Prompt Caching: 5min vs 1hr Pricing (worked 45-min example: $2.0025 vs $0.66) | https://blog.sandbase.ai/anthropic-cache-pricing-5m-1h-explained/ | 2026-08-01 | §8.3, §8.11(4) |
| Prompt Caching Guide: Cut API Costs by 90% (min-token table; Gemini 32,768; batch stacking) | https://aioutlooks.com/prompt-caching-guide/ | 2026-05-08 | §8.7(2) |
| PromptHub: Prompt Caching across providers (Gemini 75% read discount, 32,768-token min) | https://www.prompthub.us/blog/prompt-caching-with-openai-anthropic-and-google-models | 2025-10-23 | §8.7(2) |
| Helicone: Prompt Caching concepts (Vertex/Bedrock 5-min-only limitation) | https://docs.helicone.ai/gateway/concepts/prompt-caching | n.d. | §8.7(2) |
| The One Thing That Makes OpenAI 80% Faster (cache isolated at org level; only inputs cached, never outputs) | https://sgryt.com/posts/openai-prompt-caching-cost-optimization/ | 2025-05-06 | §8.3(1) |
| Prompt Caching Explained: cost-caching-rate-limits (serverless cold starts spaced > TTL ⇒ cache never hits) | https://ai-tldr.dev/learn/llm-apis/cost-caching-rate-limits/prompt-caching-explained/ | 2026-06-12 | §8.7(2) |
| Prompt Caching Infrastructure (vLLM APC 10× cost difference; break-even at 1.4 reads) | https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025 | 2026-03-17 | §8.1 self-hosted row |
| Anthropic Prompt Caching Saves 90% — the caveat (break-even ≈3 reads per write at 5-min TTL; traffic-shape dependence) | https://dev.to/gabrielanhaia/anthropic-prompt-caching-saves-90-heres-the-one-caveat-nobody-mentions-258k | 2026-04-29 | §8.3 |

### Contradiction logged this pass [X]

**Anthropic 1-hour cache-write multiplier: 2.0× or 1.5×?** intuitionlabs (2026-09-05), technspire
(2026-09-06), ofox (2026-06-10), gingerlabs and aioutlooks all say **2.0×**. blog.sandbase.ai
(2026-08-01) says **1.5×** and builds its worked example on 1.5×. Four-to-one in favour of **2.0×**;
`verify_08_switch_cost.py` reproduces sandbase's worked example using its own 1.5× figure, so the
example is arithmetically sound *given its premise* but the premise is likely wrong. **Use 2.0× in
the price registry and verify against the live Anthropic pricing page before shipping.** Added to §D.

---

## H. Sources added for the TTL-tier pass (Thread 9), verified 2026-09-12

TTL behaviour was checked against **primary documentation** this pass, not blogs, because the whole
`09` design rests on one sentence about refresh semantics.

| Source | URL | Date | Used for |
|---|---|---|---|
| **Prompt caching — Claude Platform Docs [P]** (5-min lifetime "refreshed for no additional cost each time the cached content is used"; lifetime measured from **start** of request; mixing TTLs + ordering constraint; A/B/C billing partition; `thinking`/`tool_choice`/`output_config.effort` invalidators; batch best-effort + "1-hour cache can help"; 1h platform availability; `max_tokens: 0` pre-warm) | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching | docs page (2023-06-01 stamp, content current) | §9.1, §9.3, §9.4, §9.6 |
| Prompt caching — Claude Platform Docs (alt host) | https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-04-02 | §9.1 corroboration |
| **Prompt caching for faster model inference — Amazon Bedrock [P]** (1h TTL via `cachePoint`/`cache_control`; supported-model list) | https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html | 2026-06-02 | §9.8 — **corrects the "Bedrock is 5-min only" claim in `08` §8.1** |
| **Prompt caching — Google Cloud, partner models / Claude [P]** | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching | 2026-04-21 | §9.8 |
| Anthropic Prompt Cache TTL + Cost Mechanics (1h-vs-5m write ratio is **1.6× not 2×**; sliding-window refresh on both tiers; what invalidates; Claude Code locks the tool list at startup) | https://brandonwie.dev/posts/anthropic-prompt-cache-ttl | 2026-05-31 | §9.2, §9.6, §9.12(2) |
| How Prompt Caching Actually Works in Claude Code (KV tensors in VRAM; "each cache hit resets the timer… active session keeps the cache warm indefinitely"; auto-caching; measured 90% vs 96% hit rates) | https://www.claudecodecamp.com/p/how-prompt-caching-actually-works-in-claude-code | 2026-08-21 | §9.1 |
| Prompt Caching With the Claude API: A Practical Guide (5m pays off on 2nd request, 1h on 3rd; "use 1h only when the prefix lives longer than 10 min between calls"; dual 5m/1h entries reported separately in `usage`) | https://dev.to/thegdsks/prompt-caching-with-the-claude-api-a-practical-guide-14ce | 2026-04-29 | §9.2 |
| **Bedrock: Prompt caching TTL hardcoded to 5m · anthropics/claude-code #32671** (proxy logs: $11.25 in 5m rewrites vs $5.57 in 1h reads over 100+ requests) | https://github.com/anthropics/claude-code/issues/32671 | 2026-03-10 | §9.9 — **figure audited: reads-only, honest number is 42%** |
| Claude Code 2.1.108+ 1h prompt caching fix (`ENABLE_PROMPT_CACHING_1H=1`, `FORCE_PROMPT_CACHING_5M`; practitioner quoting the primary doc on refresh) | https://www.reddit.com/r/Anthropic/comments/1slo0iu/claude_code_21108_1h_prompt_caching_fix_stop/ | 2026-04-14 | §9.1, §9.9 |
| **Prompt Caching — OpenRouter docs [P-adjacent]** (`prompt_cache_breakpoint`, `prompt_cache_options` `mode: explicit` + `ttl: "30m"`; **"TTLs are not translated"**; block markers interchangeable but TTL dropped toward OpenAI; batch guidance) | https://openrouter.ai/docs/guides/best-practices/prompt-caching | 2026-07-12 | §9.8, §9.4(2) |
| Prompt Caching — Portkey docs (TTL table; **"you can mix both TTLs in the same request, but 1-hour entries must appear before 5-minute entries"**) | https://portkey.ai/docs/integrations/llms/anthropic/prompt-caching | n.d. | §9.3 |
| Claude Prompt Caching — AIHubMix (mixed-TTL worked example; "writing the cache but never reading it" root cause) | https://docs.aihubmix.com/en/api/Claude-Cache | 2026-06-01 | §9.3 |
| **spring-projects/spring-ai #4325** — message-type-aware caching: "SYSTEM & TOOLS at 1h; USER & ASSISTANT at 5m" | https://github.com/spring-projects/spring-ai/issues/4325 | 2025-09-05 | §9.3 — independent convergence on the same policy |
| **router-for-me/CLIProxyAPI #3398** — default 5m blocks silently downgrade later `ttl: "1h"`; tools→system→messages evaluation order; max-4 enforcement | https://github.com/router-for-me/CLIProxyAPI/issues/3398 | 2026-05-14 | §9.3 — the ordering bug to test against |
| letta-ai/letta #3307 — configurable Anthropic cache TTL per-agent and global | https://github.com/letta-ai/letta/issues/3307 | 2026-04-14 | §9.7 |
| Anthropic Prompt Caching Guide (TTL "refreshed on every cache hit… active agent loop can keep the cache warm indefinitely"; per-model minimums 1024/2048) | https://prismix.dev/guides/anthropic-prompt-caching-guide | n.d. | §9.1 |
| What Is Prompt Caching in Claude Code (TTL from last access, hit resets clock; expiry = one-time cost per event, not compounding) | https://www.mindstudio.ai/blog/prompt-caching-claude-code-token-savings | 2026-05-25 | §9.1, §9.2 |
| Anthropic — Pydantic AI docs (`anthropic_cache='1h'`, per-block vs top-level caching, gateway compatibility) | https://pydantic.dev/docs/ai/models/anthropic/ | 2026-04-16 | §9.7 implementation reference |

### Corrections logged this pass

1. **`08` §8.1's "Vertex AI and Bedrock only support 5-minute caching" (via Helicone) is outdated.**
   AWS's own docs list 1h TTL support for 10 Claude models. See §9.8.
2. **`08` §8.7(3)'s "batch is exempt from cache logic" is too strong.** Exempt from session affinity,
   not from cache policy. See §9.4(2).
3. **The "1h cache is 2× more expensive" framing is comparison-frame-dependent.** 2.0× vs *uncached*;
   **1.6×** vs the 5m write. Cost models using 2× overstate the upgrade by 60%.
4. **claude-code #32671's "~50% saving" counts reads only.** Including the 1h write gives **42%**.
   Cite ~40–50%.

---

## I. Sources added for the Fusion / delegation pass (Thread 10), read 2026-09-12

| Source | URL | Date | Used for |
|---|---|---|---|
| **Introducing Fusion in Devin Desktop & CLI — Cognition [V]** (full text read, both chunks). Lead/sidekick delegation; "Model routing is not enough"; FrontierCode turn distribution; price-per-task; per-pair harness tuning; 5-benchmark cost/quality table | https://cognition.com/blog/local-fusion | 2026 (launch post, undated) | all of Thread 10 |
| Devin Fusion (earlier post, linked from the above as the "sidekick" origin) | https://cognition.com/blog/devin-fusion | not fetched | background |
| Making Fable cheaper than Opus (the Opus→Fable lead swap evidence) | https://cognition.com/blog/making-fable-cheaper-than-opus | not fetched | §10.4 |

**Confidence:** vendor primary source. Cognition describing their own product, with Artificial Analysis
and Vals AI named as evaluation partners. Better than pure self-evaluation; **no independent
replication exists.** The chart comparisons are harness-vs-harness, so architecture and harness are
confounded. Numbers are quoted as reported, not independently verified.
