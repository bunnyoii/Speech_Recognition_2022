# Meeting Minutes Assistant

## Environment Configuration

Navigate to the project root directory:

```bash
cd Meeting_Minutes_Assistant
```

Set up the backend environment:

```bash
cd backend
pip install -r requirements.txt
```

**Note**: The backend relies on PyTorch for AI model inference. Ensure that your system has a compatible version of PyTorch installed. If not, you can install it using the following command:

```bash
pip install torch
```

Set up the frontend environment:

```bash
cd frontend
cnpm install
```

## Running the Program

Run the backend application:

```bash
python main.py [--options]
```

Run the frontend application:

```bash
npm run dev
```