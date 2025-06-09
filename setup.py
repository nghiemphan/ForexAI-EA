# setup.py
"""
Automated Setup Script for ForexAI-EA Project
Automates the installation and configuration process
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-08
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json

class ForexAISetup:
    """Automated setup for ForexAI-EA project"""
    
    def __init__(self):
        self.project_root = Path("C:/ForexAI-EA")
        self.success_steps = []
        self.failed_steps = []
    
    def run_setup(self):
        """Run complete setup process"""
        print("=" * 60)
        print("🚀 ForexAI-EA Automated Setup")
        print("=" * 60)
        
        steps = [
            ("Creating project structure", self.create_project_structure),
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Creating configuration files", self.create_config_files),
            ("Setting up logging", self.setup_logging),
            ("Creating startup scripts", self.create_startup_scripts),
            ("Configuring firewall", self.configure_firewall),
            ("Running system checks", self.run_system_checks),
            ("Testing installation", self.test_installation)
        ]
        
        for step_name, step_function in steps:
            print(f"\n🔧 {step_name}...")
            try:
                step_function()
                self.success_steps.append(step_name)
                print(f"   ✅ {step_name} completed successfully")
            except Exception as e:
                self.failed_steps.append((step_name, str(e)))
                print(f"   ❌ {step_name} failed: {e}")
        
        self.print_setup_summary()
    
    def create_project_structure(self):
        """Create complete project folder structure"""
        folders = [
            "src/python",
            "src/mql5",
            "config",
            "data/historical",
            "data/models", 
            "data/logs",
            "tests",
            "docs",
            "backups"
        ]
        
        # Create main project directory
        self.project_root.mkdir(exist_ok=True)
        
        # Create subdirectories
        for folder in folders:
            (self.project_root / folder).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_files = [
            "src/python/__init__.py",
            "src/__init__.py"
        ]
        
        for init_file in init_files:
            (self.project_root / init_file).touch()
    
    def install_python_dependencies(self):
        """Install required Python packages"""
        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "pyyaml>=6.0",
            "requests>=2.26.0",
            "websocket-client>=1.2.0",
            "joblib>=1.1.0",
            "ta>=0.7.0"
        ]
        
        # Create requirements.txt
        requirements_path = self.project_root / "requirements.txt"
        with open(requirements_path, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        # Install packages
        for package in requirements:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"   Warning: Failed to install {package}: {e}")
    
    def create_config_files(self):
        """Create configuration files"""
        
        # settings.yaml
        settings_config = {
            'project': {
                'name': 'ForexAI-EA',
                'version': '1.0.0',
                'environment': 'development'
            },
            'server': {
                'host': 'localhost',
                'port': 8888,
                'timeout': 30,
                'max_connections': 10
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'data/logs/forexai_ea.log',
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'ai_models': {
                'technical_indicators': {
                    'enabled': True,
                    'weight': 0.4,
                    'model_type': 'RandomForest'
                }
            }
        }
        
        # trading_config.yaml
        trading_config = {
            'account': {
                'initial_balance': 1000,
                'currency': 'USD'
            },
            'risk_management': {
                'risk_per_trade': 0.015,
                'max_positions': 4,
                'max_daily_loss': 0.05
            },
            'trading_pairs': {
                'majors': [
                    {'symbol': 'EURUSD', 'enabled': True, 'risk_multiplier': 1.0},
                    {'symbol': 'GBPUSD', 'enabled': True, 'risk_multiplier': 1.0},
                    {'symbol': 'USDJPY', 'enabled': True, 'risk_multiplier': 1.0}
                ]
            }
        }
        
        # Write configuration files
        import yaml
        
        with open(self.project_root / "config/settings.yaml", 'w') as f:
            yaml.dump(settings_config, f, default_flow_style=False)
        
        with open(self.project_root / "config/trading_config.yaml", 'w') as f:
            yaml.dump(trading_config, f, default_flow_style=False)
    
    def setup_logging(self):
        """Setup logging directories and initial log files"""
        log_dir = self.project_root / "data/logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create initial log files
        log_files = [
            "ai_server.log",
            "trading_engine.log", 
            "error.log",
            "performance.log"
        ]
        
        for log_file in log_files:
            (log_dir / log_file).touch()
    
    def create_startup_scripts(self):
        """Create startup scripts for Windows"""
        
        # AI Server startup script
        ai_server_script = f"""@echo off
echo Starting ForexAI-EA AI Server...
cd /d "{self.project_root}\\src\\python"
python socket_server.py
pause
"""
        
        with open(self.project_root / "start_ai_server.bat", 'w') as f:
            f.write(ai_server_script)
        
        # Test communication script
        test_script = f"""@echo off
echo Testing ForexAI-EA Communication...
cd /d "{self.project_root}"
python test_communication.py
pause
"""
        
        with open(self.project_root / "test_communication.bat", 'w') as f:
            f.write(test_script)
        
        # Setup script
        setup_script = f"""@echo off
echo Running ForexAI-EA Setup...
cd /d "{self.project_root}"
python setup.py
pause
"""
        
        with open(self.project_root / "run_setup.bat", 'w') as f:
            f.write(setup_script)
    
    def configure_firewall(self):
        """Configure Windows firewall (requires admin privileges)"""
        try:
            # Try to add firewall rule for port 8888
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                "name=ForexAI-EA Server", "dir=in", "action=allow",
                "protocol=TCP", "localport=8888"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("   Warning: Could not configure firewall automatically")
                print("   Please manually allow port 8888 in Windows Firewall")
        except Exception as e:
            print(f"   Warning: Firewall configuration failed: {e}")
    
    def run_system_checks(self):
        """Run system requirement checks"""
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python version", False, "Requires Python 3.8+"))
        
        # Check disk space
        import shutil
        free_space = shutil.disk_usage(self.project_root.parent).free / (1024**3)  # GB
        if free_space >= 5:
            checks.append(("Disk space", True, f"{free_space:.1f} GB available"))
        else:
            checks.append(("Disk space", False, f"Only {free_space:.1f} GB available"))
        
        # Check network connectivity
        try:
            urllib.request.urlopen('http://www.google.com', timeout=10)
            checks.append(("Internet connection", True, "Connected"))
        except:
            checks.append(("Internet connection", False, "No connection"))
        
        # Print check results
        for check_name, success, message in checks:
            status = "✅" if success else "❌"
            print(f"   {status} {check_name}: {message}")
    
    def test_installation(self):
        """Test the installation"""
        # Check if all required files exist
        required_files = [
            "config/settings.yaml",
            "config/trading_config.yaml", 
            "start_ai_server.bat",
            "test_communication.bat"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise Exception(f"Missing files: {', '.join(missing_files)}")
        
        # Test Python imports
        test_imports = [
            "numpy", "pandas", "sklearn", "yaml"
        ]
        
        for module in test_imports:
            try:
                __import__(module)
            except ImportError:
                raise Exception(f"Cannot import required module: {module}")
    
    def print_setup_summary(self):
        """Print setup completion summary"""
        print("\n" + "=" * 60)
        print("🎯 SETUP SUMMARY")
        print("=" * 60)
        
        print(f"✅ Successful steps: {len(self.success_steps)}")
        for step in self.success_steps:
            print(f"   ✅ {step}")
        
        if self.failed_steps:
            print(f"\n❌ Failed steps: {len(self.failed_steps)}")
            for step, error in self.failed_steps:
                print(f"   ❌ {step}: {error}")
        
        print(f"\n📁 Project created at: {self.project_root}")
        
        if not self.failed_steps:
            print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("\nNext steps:")
            print("1. Copy socket_server.py to src/python/")
            print("2. Copy ForexAI_EA_v1.mq5 to src/mql5/")
            print("3. Run start_ai_server.bat to start AI server")
            print("4. Run test_communication.bat to test connection")
            print("5. Open MetaTrader 5 and attach the EA")
        else:
            print("\n⚠️  SETUP COMPLETED WITH ERRORS")
            print("Please fix the failed steps before proceeding")

def main():
    """Main setup function"""
    print("Welcome to ForexAI-EA Automated Setup!")
    print("This will install and configure the trading system.")
    
    # Check if running as administrator
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if not is_admin:
        print("\n⚠️  Warning: Not running as administrator")
        print("Some features (like firewall configuration) may fail")
    
    response = input("\nProceed with setup? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled")
        return
    
    # Run setup
    setup = ForexAISetup()
    setup.run_setup()

if __name__ == "__main__":
    main()