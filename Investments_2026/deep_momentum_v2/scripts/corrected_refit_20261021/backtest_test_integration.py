import copy
import datetime
import unittest
import backtest_integration as b


def windows():
    rows=[]
    for year in b.YEARS:
        for month in range(1,13):
            first=datetime.date(year,month,1);previous=first-datetime.timedelta(days=1)
            last=(datetime.date(year+1,1,1) if month==12 else datetime.date(year,month+1,1))-datetime.timedelta(days=1)
            rows.append(dict(signal_date=previous.isoformat(),outcome_month=f'{year}-{month:02}',outcome_start=first.isoformat(),outcome_end=last.isoformat()))
    return rows


def forecasts(year=2020):
    return [dict(source_security_id='source'+str(i),intrinio_security_id='native'+str(i),
        **w,identity_eligible=True,paper_prediction_eligible=True,prediction_eligible=True,prediction_status='predicted',
        fit_year=year,training_outcome_start=f'{year-10}-01-01',training_outcome_end=f'{year-1}-12-31',
        training_label_observed_through={2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}[year])
        for i,w in enumerate(windows()) if w['outcome_month'].startswith(str(year))]


class Integration(unittest.TestCase):
    def test_120_exact_scenarios(self):
        self.assertEqual(len(b.SCENARIOS),120);self.assertEqual(len(set(b.SCENARIOS)),120)
    def test_prior_ten_years_and_twelve_months(self):
        self.assertEqual(len(b.forecasts_metadata(forecasts(),2020,windows())),12)
    def test_future_label_never_required_for_admission(self):
        rows=forecasts();rows[0]['target_available']=False
        self.assertEqual(len(b.forecasts_metadata(rows,2020,windows())),12)
    def test_xom_typed_key_preserved(self):
        rows=forecasts();rows[0]['source_security_id']=b.XOM_SOURCE;rows[0]['intrinio_security_id']=b.XOM_KEY
        b.forecasts_metadata(rows,2020,windows())
        rows[0]['intrinio_security_id']='sec_z5YpGE'
        with self.assertRaises(ValueError):b.forecasts_metadata(rows,2020,windows())
    def test_bk_native_preserved(self):
        rows=forecasts();rows[0]['source_security_id']=b.BK_SOURCE;rows[0]['intrinio_security_id']=b.BK_KEY
        b.forecasts_metadata(rows,2020,windows())
        rows[0]['intrinio_security_id']='sec_gA3NYQ'
        with self.assertRaises(ValueError):b.forecasts_metadata(rows,2020,windows())
    def test_changed_fit_window_fails(self):
        rows=forecasts();rows[0]['training_outcome_start']='2000-01-01'
        with self.assertRaises(ValueError):b.forecasts_metadata(rows,2020,windows())
    def test_duplicate_signal_source_fails(self):
        rows=forecasts();rows.append(rows[0].copy())
        with self.assertRaises(ValueError):b.forecasts_metadata(rows,2020,windows())
    def test_changed_prediction_admission_fails(self):
        rows=forecasts();rows[0]['paper_prediction_eligible']=False
        with self.assertRaises(ValueError):b.forecasts_metadata(rows,2020,windows())
    def test_source_date_projection_does_not_use_holdings(self):
        sessions=[(datetime.date(2020,1,1)+datetime.timedelta(days=i)).isoformat() for i in range(1006)]
        rows=[dict(intrinio_security_id='arbitrary',trading_date=d) for d in ['2019-12-31','2020-01-02','2024-01-02']]
        positions,outside=b.scoped_quote_positions(rows,2020,sessions)
        self.assertEqual(positions,[1]);self.assertEqual(len(outside),2)
    def test_duplicate_quote_keys_fail(self):
        sessions=[(datetime.date(2020,1,1)+datetime.timedelta(days=i)).isoformat() for i in range(1006)];rows=[dict(intrinio_security_id='n',trading_date='2020-01-02')]*2
        with self.assertRaises(ValueError):b.scoped_quote_positions(rows,2020,sessions)
    def test_accounting_projection_has_no_volumes(self):
        self.assertEqual(len(b.ACTION_COLUMNS),18);self.assertNotIn('volume',b.ACTION_COLUMNS);self.assertNotIn('adj_volume',b.ACTION_COLUMNS)

if __name__=='__main__':unittest.main()
