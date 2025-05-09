# Semiasynchronous Federated Learning (FL) over Flwr Framework - PhD Research

---

## Introduction

This repository contains the implementation of a **Semiasynchronous Federated Learning (FL)** system using the well-known **Flwr framework** for Federated Learning. The main goal of this research is to explore and implement **semiasynchronous behavior** within FL systems to improve **communication efficiency** and **convergence speed** while maintaining client data privacy.

Federated Learning is a decentralized approach where clients collaboratively train a shared model without centralizing their data. Traditional FL requires synchronized updates, but this research aims to enable more flexibility by allowing clients to send updates asynchronously, which can reduce idle times and improve scalability.

---

## Project Objectives
- Implement **semiasynchronous FL** within the **Flwr framework**.
- Investigate the impact of **asynchronous updates** on model performance and communication efficiency.
- Experiment with different **configurations** and **federation modes** (local-execution vs. remote-execution).
- Assess the effect of **client data distributions** (iid vs noniid) on training performance.

---

## Get Started

Follow the steps below to set up the project on your local machine.

### 0. Prerequisites
- Docker from official guide and being able to execute containers with non-root user.
- Python3 and Python3 venv packages.
- Git package.

### 1. Clone the repository
```bash
git clone https://github.com/VictorHidalgoUCLM/FL_Semiasync.git
```

### 2. Navigate to the cloned directory and create a new Python virtual environment
```bash
cd FL_Semiasync
python -m venv Flwr

source Flwr/bin/activate
```

### 3. Install project dependencies
```bash
pip install -e .
```

### 4. Decide on the execution mode and run the experiment

You can configure the execution parameters as follows:

- **Federation Mode**: Choose between `local-execution` or `remote-execution`.
- **Synchronization Type (`s`)**: Specify the number of clients in sync or the sync type (M value).
- **Number of Executions (`n`)**: Define how many executions to run per configuration.
- **Debug Mode (`d`)**: Enable debug mode for detailed logging.
- **Data Type (`t`)**: Choose between `iid` (independent and identically distributed) or `noniid` (non-iid) data.

To run the experiment with your desired configurations, execute:

```bash
python run.py --federation <federation_mode> --s <sync_type> --n <num_executions> --d <debug_mode> --t <data_type>
```

#### Example:
```bash
python run.py --federation local-execution --s 10 --n 5 --d True --t iid
```
