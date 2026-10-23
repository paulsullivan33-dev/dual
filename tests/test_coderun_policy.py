"""Docker selection is explicit, restricted, and never silently downgraded."""
import os
import unittest
from unittest.mock import patch

import ollama_coderun as cr


class ContainerPolicyTests(unittest.TestCase):
    def test_container_restrictions_and_no_host_credentials_or_mounts(self):
        with patch.object(cr.shutil, 'which', return_value='/tools/docker'), \
             patch.dict(os.environ, {'SECRET_TOKEN': 'must-not-forward'}):
            command = cr._container_command('duel-run-test', ['--test'])
        for flag in ('--pull=never', '--read-only', '--rm'):
            self.assertIn(flag, command)
        for key, value in (('--network', 'none'), ('--cap-drop', 'ALL'),
                           ('--security-opt', 'no-new-privileges'), ('--memory', '128m'),
                           ('--memory-swap', '128m'), ('--pids-limit', '64'),
                           ('--cpus', '1'), ('--user', '65534:65534'), ('--log-driver', 'none')):
            self.assertEqual(command[command.index(key) + 1], value)
        for flag in ('--privileged', '-v', '--volume', '--mount', '-e', '--env', '--env-file'):
            self.assertNotIn(flag, command)
        self.assertNotIn('must-not-forward', command)
        self.assertEqual(command[-1], '--test')

    def test_missing_docker_does_not_attempt_host_execution(self):
        with patch.object(cr.shutil, 'which', return_value=None), patch.object(cr.subprocess, 'Popen') as popen:
            result = cr.run_program('raise AssertionError("must not execute")', backend="docker")
        popen.assert_not_called()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Docker runner selected', result['output'])

    def test_cleanup_uses_named_container_not_host_pids(self):
        with patch.object(cr.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.assertTrue(cr._remove_container('/tools/docker', 'duel-run-test'))
        self.assertEqual(run.call_args.args[0], ['/tools/docker', 'rm', '--force', 'duel-run-test'])

    def test_invalid_timeouts_and_large_source_rejected_before_execution(self):
        with patch.object(cr.subprocess, 'Popen') as popen:
            for value in (0, -1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    cr.run_program('pass', timeout=value)
            with self.assertRaises(ValueError):
                cr.run_program('x' * (cr.MAX_CODE_BYTES + 1))
        popen.assert_not_called()
