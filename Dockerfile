# ==========================================
# Retail Demand Forecasting API — Dockerfile (Final Stable)
# Prophet + CmdStanPy + LightGBM + XGBoost + TFT support
# ==========================================

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --------------------------------------------------------
# 🧩 System dependencies for Prophet, CmdStanPy & PyStan
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    cmake \
    git \
    curl \
    wget \
    python3-setuptools \
    python3-pip \
    python3-distutils-extra \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# 📦 Copy and install Python dependencies
# --------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# ⚙️ Prophet + CmdStanPy setup (stable configuration)
# --------------------------------------------------------
# Prophet 1.1.5 is the most reliable version for CmdStanPy backend
RUN pip install --no-cache-dir prophet==1.1.5 cmdstanpy==1.2.3

# Install CmdStan backend (used by Prophet)
RUN python -m cmdstanpy.install_cmdstan --cores=2 --progress

# --------------------------------------------------------
# 🧠 Copy project files
# --------------------------------------------------------
COPY . .

# Verify contents (debugging convenience)
RUN echo "📂 Contents of /app:" && ls -lh /app

# --------------------------------------------------------
# 🌍 Expose port and launch API
# --------------------------------------------------------
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
