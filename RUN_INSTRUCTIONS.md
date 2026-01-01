# How to Run the Application

This guide will help you set up and run both the backend and frontend.

## Prerequisites

- **Python 3.8+** installed
- **Node.js and npm** installed
- **PowerShell** (Windows)

## Quick Start (Using Scripts)

### Option 1: Automated Setup
Run the setup script first to install all dependencies:
```powershell
.\setup.ps1
```

### Option 2: Manual Setup
Follow the manual steps below.

## Manual Setup

### Backend Setup

1. **Navigate to the project root directory**

2. **Create and activate a virtual environment:**
   ```powershell
   cd backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the backend server:**
   ```powershell
   uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
   ```
   
   Or use the run script from the project root:
   ```powershell
   .\run-backend.ps1
   ```

   The backend will be available at: **http://127.0.0.1:8000**
   
   API documentation will be at: **http://127.0.0.1:8000/docs**

### Frontend Setup

1. **Open a NEW terminal window** (keep the backend running in the first terminal)

2. **Navigate to the frontend directory:**
   ```powershell
   cd frontend-react
   ```

3. **Install dependencies (first time only):**
   ```powershell
   npm install
   ```

4. **Run the frontend development server:**
   ```powershell
   npm run dev
   ```
   
   Or use the run script from the project root:
   ```powershell
   .\run-frontend.ps1
   ```

   The frontend will typically be available at: **http://localhost:5173**

## Running Both Servers

You need **TWO terminal windows**:

1. **Terminal 1 - Backend:**
   ```powershell
   .\run-backend.ps1
   ```

2. **Terminal 2 - Frontend:**
   ```powershell
   .\run-frontend.ps1
   ```

## Troubleshooting

### Backend Issues

- **Port 8000 already in use:** 
  - Kill the process using port 8000, or change the port: `--port 8001`
  
- **Module not found errors:**
  - Make sure the virtual environment is activated
  - Reinstall dependencies: `pip install -r backend\requirements.txt`

- **Permission errors with PowerShell:**
  - Run PowerShell as Administrator
  - Or set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Frontend Issues

- **Port 5173 already in use:**
  - Vite will automatically try the next available port (5174, 5175, etc.)
  
- **Node modules errors:**
  - Delete `node_modules` and `package-lock.json`, then run `npm install` again

## Environment Variables

The backend may require environment variables for API keys (e.g., Google Gemini API key). Check the backend code for required environment variables.

## Stopping the Servers

Press `CTRL+C` in each terminal to stop the respective server.

