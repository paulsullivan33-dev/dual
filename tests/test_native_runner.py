"""Native runner resource behavior for small trusted scenarios."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import ollama_coderun as cr


class NativeRunnerTests(unittest.TestCase):
    def test_default_native_runner_never_looks_for_docker(self):
        with patch.object(cr, '_container_command', side_effect=AssertionError('no Docker')):
            result = cr.run_program('import math; print(math.factorial(10))')
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertIn('3628800', result['output'])

    def test_native_workspace_is_disposable(self):
        result = cr.run_program("import os; open('made.txt','w').close(); print(os.getcwd())")
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertFalse(Path(result['output'].strip()).exists())

    def test_native_program_has_file_context_and_can_import_its_own_helpers(self):
        result = cr.run_program("from pathlib import Path; "
                                "assert Path(__file__).name == 'program.py'; "
                                "Path('helper.py').write_text('answer=42'); "
                                "import helper; print(helper.answer)")
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertEqual(result['output'].strip(), '42')

    def test_native_unicode_output_is_utf8(self):
        result = cr.run_program("print('caf\u00e9 \u6f22\u5b57')")
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertEqual(result['output'].strip(), 'caf\u00e9 \u6f22\u5b57')

    def test_memory_budget_is_configurable_for_docker(self):
        with patch.object(cr.shutil, 'which', return_value='/tools/docker'):
            command = cr._container_command('test', [], memory_mb=512)
        self.assertEqual(command[command.index('--memory') + 1], '512m')
        self.assertEqual(command[command.index('--memory-swap') + 1], '512m')

    def test_invalid_backend_and_memory_budget_rejected(self):
        for backend, memory in [('unknown', 128), ('native', 0), ('docker', True), ('native', 31)]:
            with self.subTest(backend=backend, memory=memory), self.assertRaises(ValueError):
                cr.run_program('pass', backend=backend, memory_mb=memory)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux resource limits')
    def test_linux_child_limits_leave_parent_limits_unchanged(self):
        import resource  # pylint: disable=import-error
        before = resource.getrlimit(resource.RLIMIT_AS)
        result = cr.run_program("import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0]); "
                                "print(resource.getrlimit(resource.RLIMIT_CPU)[0])", timeout=3, memory_mb=64)
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertEqual(result['output'].splitlines(), [str(64 * 1024 * 1024), '3'])
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), before)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux resource limits')
    def test_linux_memory_exhaustion_is_a_child_failure(self):
        result = cr.run_program("try:\n data=bytearray(256*1024*1024)\n"
                                "except MemoryError:\n print('bounded-memory'); raise SystemExit(7)", memory_mb=64)
        self.assertEqual(result['exit_code'], 7, result['output'])
        self.assertIn('bounded-memory', result['output'])

    @unittest.skipUnless(os.name == 'posix', 'POSIX process-group cleanup')
    def test_ordinary_descendant_cannot_keep_running_after_parent_exits(self):
        result = cr.run_program("import subprocess,sys; "
                                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                                "print('parent-finished')", timeout=4)
        self.assertEqual(result['status'], 'ok', result['output'])
        self.assertLess(result['seconds'], 5)
        self.assertIn('parent-finished', result['output'])
