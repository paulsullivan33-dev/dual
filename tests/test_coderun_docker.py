"""Explicit Docker integration checks; enabled only by the dedicated CI job."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ollama_coderun as cr


@unittest.skipUnless(os.environ.get('DUAL_TEST_DOCKER') == '1', 'Dedicated Docker test job only')
class DockerIntegrationTests(unittest.TestCase):
    def test_success_and_arguments(self):
        result = cr.run_program("import sys; print(sys.argv[1:])", args=['--test'])
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertIn("['--test']", result['output'])

    def test_host_files_environment_and_network_interfaces_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            host_file = Path(directory) / 'host-only.txt'
            host_file.write_text('must remain outside the container', encoding='utf-8')
            code = ("import os; from pathlib import Path; "
                    f"assert not Path({str(host_file)!r}).exists(); "
                    "assert 'DUAL_TEST_SECRET' not in os.environ; "
                    "assert os.geteuid() == 65534; "
                    "assert set(os.listdir('/sys/class/net')) == {'lo'}; "
                    "print('isolated')")
            with patch.dict(os.environ, {'DUAL_TEST_SECRET': 'test-fixture-secret'}):
                result = cr.run_program(code)
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertIn('isolated', result['output'])

    def test_output_flood_keeps_a_bounded_tail(self):
        result = cr.run_program("print('x' * 2000000); print('final-marker')")
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertLessEqual(len(result['output']), cr.MAX_OUTPUT_CHARS + 50)
        self.assertIn('final-marker', result['output'])

    def test_timeout_removes_the_container_and_its_descendant(self):
        with patch.object(cr, '_remove_container', wraps=cr._remove_container) as cleanup:
            result = cr.run_program(
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                "time.sleep(60)", timeout=2)
        self.assertEqual(result['status'], 'timeout', result['output'])
        name = cleanup.call_args.args[1]
        inspected = subprocess.run([shutil.which('docker'), 'container', 'inspect', name],
                                   capture_output=True, timeout=10, check=False)
        self.assertNotEqual(inspected.returncode, 0, 'Timed-out container survived cleanup')
