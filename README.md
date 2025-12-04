## AI Agents as Future Managers : A Toy Reinforcement Learning Study 

### Project Overview 

This repository contains the final project for a machine learning course, exploring the question:

> **Can AI agents eventually replace or augment human managerial roles?**

The project combines:
- A **conceptual report** on future AI capabilities and organizational impact  
- A **toy reinforcement learning (RL) environment** where an AI “manager” learns to assign tasks to AI “workers”

---

### Repository Structure


```
.
├── AI_Agent.py                         
├── curve.png                           
├── report.md                           
└── README.md                          
```

---

### Concept & Motivation

In future organizational ecosystems, a plausible configuration may involve:

- A single human supervisor  
- Overseeing a group of autonomous **AI Agents**

These AI Agents may take charge of functions such as:

- Human resources (scheduling, performance monitoring)  
- Customer service (automated responses and conversational assistance)  
- Marketing (advertising optimization)  
- Data analytics (decision support)  
- Resource and workflow management (efficient allocation and coordination)

The objective of this project is *not* to construct a real-world AI management system.  
Instead, through a **controlled and reproducible toy model**, we aim to address a smaller but fundamental question:

> If an AI manager is allowed to learn “how to assign tasks of different difficulties to workers with different capabilities,”  
> **can it acquire a reasonable—or even optimal—allocation strategy?**

---

### Toy Model Overview 

#### Problem Setting

- Manager agent × 1  
- Worker agents × 3, each with fixed capability  
- Tasks × 5, each with a defined difficulty level

The goal of the AI manager is:  
**to learn how to match tasks to the most appropriate worker in order to maximize the overall success rate.**

---

### Reinforcement Learning Setup

#### State 
- Task difficulty  
- Capability levels of the three workers  
- Current task index

#### Action
- Select one of the three workers to handle the task

#### Reward 
- Success: +1  
- Failure: 0

#### Algorithm
- Policy Gradient (REINFORCE)  
- Policy Network: a simple fully connected neural network

---

### Running the Code

1. Create a virtual environment (optional)：

```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies：

```
pip install numpy torch matplotlib
```

3. Run：

```
python AI_Agents.py
```

Expected output：

```
[Evaluate] average tasks completed per episode = x.00 / 5
```

A plot `training_curve.png` will also be shown.

---

### Results

#### Baseline
- Average performance: **2.5 / 5**  
- No strategy — purely random allocation

#### After Reinforcement Learning
- Achieves **5.00 / 5** — a perfect strategy  
- The AI manager successfully learns the underlying rules:
  - Easy tasks → assign to any worker  
  - Medium tasks → assign to medium or strong worker  
  - Hard tasks → always assign to the strongest worker

---

### Relation to AI Replacing Human Work 

Although highly simplified, this toy model demonstrates:

- AI can learn resource allocation policies through trial and error  
- This capability generalizes to potential future AI managerial roles  
- It illustrates that AI could not only automate low-level tasks,  
  but also **partially replace or augment managerial decision-making**

For the full discussion, please refer to:

- `report.md`

---

### Author

- Course : Machine Learning  
- Topic : AI Agents as Future Managers  
- Student : *Sam Hung*  
- Institution : *Mathematical Modeling and Scientific Computing / NYCU*  

Feel free to explore, extend, or improve the code and experiments!
