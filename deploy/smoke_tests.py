"""Smoke tests and health checks for deployed model."""

import argparse
import logging
import time
import base64
from io import BytesIO

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentHealthCheck:
    """Health check and smoke tests for deployed model."""
    
    def __init__(self, base_url: str = 'http://localhost:8000', timeout: int = 10):
        """Initialize health checker.
        
        Args:
            base_url: API base URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.results = []
    
    def test_health_endpoint(self) -> bool:
        """Test health endpoint.
        
        Returns:
            True if test passed.
        """
        try:
            response = requests.get(
                f'{self.base_url}/health',
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                assert data['status'] == 'healthy'
                logger.info('✓ Health endpoint test passed')
                self.results.append(('health_check', True, None))
                return True
        except Exception as e:
            logger.error(f'✗ Health endpoint test failed: {str(e)}')
            self.results.append(('health_check', False, str(e)))
            return False
    
    def test_prediction_endpoint(self) -> bool:
        """Test prediction endpoint with sample image.
        
        Returns:
            True if test passed.
        """
        try:
            # Create sample image
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            b64_image = base64.b64encode(buffer.getvalue()).decode()
            
            response = requests.post(
                f'{self.base_url}/predict',
                json={'image': b64_image},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 'prediction' in data
                assert data['prediction'] in ['cat', 'dog']
                assert 'confidence' in data
                assert 0 <= data['confidence'] <= 1
                
                logger.info(
                    f'✓ Prediction endpoint test passed: '
                    f'{data["prediction"]} (confidence: {data["confidence"]:.4f})'
                )
                self.results.append(('prediction', True, None))
                return True
        except Exception as e:
            logger.error(f'✗ Prediction endpoint test failed: {str(e)}')
            self.results.append(('prediction', False, str(e)))
            return False
    
    def test_info_endpoint(self) -> bool:
        """Test info endpoint.
        
        Returns:
            True if test passed.
        """
        try:
            response = requests.get(
                f'{self.base_url}/info',
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 'name' in data
                assert 'version' in data
                assert 'classes' in data
                
                logger.info(f'✓ Info endpoint test passed: {data["name"]} v{data["version"]}')
                self.results.append(('info', True, None))
                return True
        except Exception as e:
            logger.error(f'✗ Info endpoint test failed: {str(e)}')
            self.results.append(('info', False, str(e)))
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test metrics endpoint.
        
        Returns:
            True if test passed.
        """
        try:
            response = requests.get(
                f'{self.base_url}/metrics',
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 'total_requests' in data
                assert 'total_predictions' in data
                
                logger.info(
                    f'✓ Metrics endpoint test passed: '
                    f'Requests: {data["total_requests"]}, '
                    f'Predictions: {data["total_predictions"]}'
                )
                self.results.append(('metrics', True, None))
                return True
        except Exception as e:
            logger.error(f'✗ Metrics endpoint test failed: {str(e)}')
            self.results.append(('metrics', False, str(e)))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests.
        
        Returns:
            True if all tests passed.
        """
        logger.info('Starting smoke tests...')
        logger.info(f'Target: {self.base_url}')
        
        tests = [
            self.test_health_endpoint,
            self.test_info_endpoint,
            self.test_metrics_endpoint,
            self.test_prediction_endpoint
        ]
        
        results = [test() for test in tests]
        
        logger.info('\n=== Test Summary ===')
        for test_name, passed, error in self.results:
            status = '✓ PASS' if passed else '✗ FAIL'
            logger.info(f'{status}: {test_name}')
            if error:
                logger.info(f'  Error: {error}')
        
        passed_count = sum(1 for _, p, _ in self.results if p)
        total_count = len(self.results)
        logger.info(f'\nTotal: {passed_count}/{total_count} tests passed')
        
        return all(results)


def main():
    """Run smoke tests."""
    parser = argparse.ArgumentParser(
        description='Run smoke tests for deployed model'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=10,
        help='Request timeout in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    health_check = DeploymentHealthCheck(
        base_url=args.url,
        timeout=args.timeout
    )
    
    success = health_check.run_all_tests()
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
