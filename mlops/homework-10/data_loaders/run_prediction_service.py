if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import subprocess
import os
import sys
import socket
import time

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port=9696, max_port=9796):
    for port in range(start_port, max_port):
        if not is_port_in_use(port):
            return port
    raise Exception("No free ports found in range")

@data_loader
def run_prediction_service(*args, **kwargs):
    """
    Запускает Flask сервис предсказаний
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(current_dir, 'predict.py')
        
        if not os.path.exists(predict_script):
            raise FileNotFoundError(f"predict.py not found at {predict_script}")
        port = find_free_port()
        print(f"Using port: {port}")
        process = subprocess.Popen(
            [sys.executable, predict_script, str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(2)
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise Exception(f"Process failed to start. Error: {stderr.decode()}")
        
        print(f"Prediction service started with PID: {process.pid} on port {port}")
        
        return {
            'status': 'Prediction service started',
            'pid': process.pid,
            'port': port
        }
        
    except Exception as e:
        print(f"Error starting prediction service: {str(e)}")
        raise

@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert 'status' in output, 'No status in output'
    assert 'pid' in output, 'No process ID in output'
    assert 'port' in output, 'No port in output'