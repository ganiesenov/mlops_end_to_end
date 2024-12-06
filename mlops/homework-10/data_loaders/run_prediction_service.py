if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import subprocess
import os
import sys
import socket
import time
import logging
import psutil
import signal

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXED_PORT = 9704

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            for conns in psutil.net_connections(kind='inet'):
                if conns.laddr.port == port and conns.pid:
                    process = psutil.Process(conns.pid)
                    process.terminate()
                    process.wait(timeout=3)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    return False

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

@data_loader
def run_prediction_service(*args, **kwargs):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(current_dir, 'predict.py')
        
        print(f"Current directory: {current_dir}")
        print(f"Predict script path: {predict_script}")
        
        # Устанавливаем необходимые зависимости
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-cors", "psutil"])
        
        # Проверяем порт и убиваем процесс если он занят
        if is_port_in_use(FIXED_PORT):
            print(f"Port {FIXED_PORT} is in use. Attempting to kill existing process...")
            if kill_process_on_port(FIXED_PORT):
                print("Successfully killed existing process")
                time.sleep(2)  # Даем время на освобождение порта
            else:
                print("Could not kill existing process")
        
        # Запускаем новый процесс
        process = subprocess.Popen(
            [sys.executable, predict_script, str(FIXED_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(2)
        if process.poll() is not None:
            returncode = process.poll()
            stdout, stderr = process.communicate()
            print(f"Process exited with code {returncode}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            raise Exception(f"Process failed to start with code {returncode}")
            
        print(f"Prediction service started with PID: {process.pid} on port {FIXED_PORT}")
        
        return {
            'status': 'Prediction service started',
            'pid': process.pid,
            'port': FIXED_PORT
        }
        
    except Exception as e:
        print(f"Error in run_prediction_service: {str(e)}")
        raise

@test
def test_output(output, *args) -> None:
    """
    Проверяет корректность запуска сервиса
    """
    assert output is not None, 'The output is undefined'
    assert 'status' in output, 'No status in output'
    assert 'pid' in output, 'No process ID in output'
    assert 'port' in output, 'No port in output'