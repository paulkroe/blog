---
title: ""
---

<style>

/* quick page-scoped styles — move to your theme CSS later */
.projects-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.25rem;
  margin-top: 1.5rem;
}
.project-card {
  display: flex;
  flex-direction: column;
  text-decoration: none;
  color: inherit;
  background: var(--page-bg); /* matches theme */
  border: 3px solid #0d1b26;
  border-radius: 10px;
  padding: 1.25rem;
  align-items: center;
  transition: transform .18s ease, box-shadow .18s ease;
  box-shadow: 0 4px 0 rgba(13,27,38,0.06);
  padding: 1.5rem;
}
.project-card:hover {
  transform: translateY(-6px);
  border-color: var(--light-secondary-color); /* light accent on hover */
  box-shadow: 0 12px 30px rgba(2,16,26,0.12);
}
.card-media {
  width: 100%;
  max-width: 300px;
  margin-bottom: 1rem;
  display: flex;
  justify-content: center;
  align-items: center;
}

.card-media img {
  width: 100%;
  max-height: 250px;
  object-fit: contain;
  border-radius: 8px;
  background: transparent;
}

.project-card h3 {
  margin: 0.25rem 0 0.35rem;
  font-size: 1.25rem;
  text-align: center;
}
.project-card p {
  margin: 0;
  font-size: 0.8rem;
  text-align: center;
  color: inherit;
  font-weight: 600;
}
</style>


<div class="projects-grid">

  <a class="project-card" href="/projects/en/dqn-atari/">
    <div class="card-media">
      <img src="/gifs/dqn_epoch_500.gif" alt="Atari Breakout Preview" loading="lazy">
    </div>
    <h3>DQN for Atari Breakout</h3>
    <p>From scratch implementation of some iconic RL techniques for solving Atari Games</p>
  </a>

  <a class="project-card" href="/projects/en/grpo/">
    <div class="card-media">
      <img src="/images/grpo-reward.png" alt="GRPO Preview" loading="lazy">
    </div>
    <h3>Mathematical Reasoning with GRPO</h3>
    <p>RL-based post-training method for increasing reasoning capabilities in LLMs</p>
  </a>
  
  <a class="project-card" href="/projects/en/bias-auditing/">
    <div class="card-media">
      <img src="/images/bias-auditing.jpg" alt="Bias Auditing Preview" loading="lazy">
    </div>
    <h3>Don't Change My View!</h3>
    <p>Ideological Bias Auditing in LLMs</p>
  </a>

  <a class="project-card" href="/projects/en/streaminator/">
    <div class="card-media">
      <img src="/images/streaminator.png" alt="Streaminator Preview" loading="lazy">
    </div>
    <h3>Streaminator</h3>
    <p>Multi-Answer Speculative Decoding for effficent LLM Inference</p>
  </a>

  <a class="project-card" href="/projects/en/schokoban/">
    <div class="card-media">
      <img src="/images/schokoban.png" alt="Schokoban Preview" loading="lazy">
    </div>
    <h3>Schokoban</h3>
    <p>Monte Carlo Tree Search for Solving Sokoban</p>
  </a>
  
  <a class="project-card" href="/projects/en/autograd/">
    <div class="card-media">
      <img src="/images/autograd.png" alt="C++ Autograd Preview" loading="lazy">
    </div>
    <h3>C++ Autograd</h3>
    <p>Basic C++ Autograd Engine written in C++</p>
  </a>
  
  <a class="project-card" href="/projects/en/torchify/">
    <div class="card-media">
      <img src="/images/torchify.png" alt="Torchify Preview" loading="lazy">
    </div>
    <h3>Torchify</h3>
    <p>Compiling a json-like file to a torch.nn.Module</p>
  </a>

</div>