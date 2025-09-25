# Configuration Guide

## Environment Variables

You can configure the Oratio backend using environment variables or by modifying `config.py` directly.

### Required Environment Variables

```bash
# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# MySQL Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=oratio

# Optional: Hugging Face model configuration
HF_MODEL_NAME=unitary/toxic-bert
```

### Setting Environment Variables

#### Windows (PowerShell)
```powershell
$env:SECRET_KEY="your-secret-key"
$env:MYSQL_PASSWORD="your_password"
```

#### Windows (Command Prompt)
```cmd
set SECRET_KEY=your-secret-key
set MYSQL_PASSWORD=your_password
```

#### Linux/macOS
```bash
export SECRET_KEY="your-secret-key"
export MYSQL_PASSWORD="your_password"
```

## MySQL Setup

1. **Install MySQL Server**
   - Windows: Download from https://dev.mysql.com/downloads/mysql/
   - Linux: `sudo apt-get install mysql-server` (Ubuntu/Debian)
   - macOS: `brew install mysql`

2. **Start MySQL Service**
   - Windows: Start MySQL service from Services
   - Linux: `sudo systemctl start mysql`
   - macOS: `brew services start mysql`

3. **Create Database**
   ```bash
   python setup_database.py
   ```

4. **Verify Connection**
   ```bash
   python test_mysql.py
   ```

## Default Configuration

If no environment variables are set, the following defaults are used:

- **Database**: `mysql+pymysql://root:@localhost:3306/oratio`
- **Secret Key**: `your-secret-key-change-in-production`
- **Token Expiry**: 30 minutes
- **Model**: `unitary/toxic-bert`
