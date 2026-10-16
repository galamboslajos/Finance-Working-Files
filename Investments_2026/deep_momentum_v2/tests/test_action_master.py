"""The action notebook stage verifies reports and clears stale state only."""
import unittest
from unittest.mock import Mock, patch
from test_master_notebook import cell_containing


class ActionMasterTests(unittest.TestCase):
    def test_failed_reader_clears_all_old_views(self):
        names=('action_backtest_run','action_backtest_comparison_df','action_backtest_repairs_df',
               'action_backtest_failures_df','action_backtest_actions_df')
        state={'catalog':Mock(),**{name:object() for name in names}}
        with patch('dmv2.action_backtest.read_action_backtest',side_effect=ValueError('stale')):
            with self.assertRaisesRegex(ValueError,'stale'):
                exec(cell_containing('action_backtest_run = read_action_backtest(catalog)'),state)
        for name in names:self.assertIsNone(state[name])

    def test_missing_saved_run_never_starts_accounting(self):
        state={'catalog':Mock()}
        with patch('dmv2.action_backtest.read_action_backtest',side_effect=FileNotFoundError), \
             patch('dmv2.action_backtest.run_action_backtest') as run, \
             patch('dmv2.action_backtest.load_action_inputs') as load, \
             patch('dmv2.action_backtest.run_action_accounting') as accounting:
            exec(cell_containing('action_backtest_run = read_action_backtest(catalog)'),state)
        run.assert_not_called();load.assert_not_called();accounting.assert_not_called()


if __name__=='__main__':unittest.main()
