"""Equal-weight both-tail stopping on fixed complete-outcome validation dates.

Predict every candidate on every authorized validation date. Outcome coverage
determines a fixed metric observation mask before the first model prediction;
it never changes stock selection or model forecasts. Unknown selected returns
remain unknown. All date/round diagnostics, including excluded dates, persist.
"""
import numpy as np
import pandas as pd
from ret_backtest import select_portfolio_indices

METRIC='mean_gross_portfolio_return_n2_n3'


class PortfolioStopping:
    def __init__(self,frame):
        required=['trade_date','security_id','future_oc_return','outcome_status']
        if not set(required)<=set(frame):raise ValueError('Missing validation candidate fields.')
        if frame.duplicated(['trade_date','security_id']).any():raise ValueError('Duplicate validation candidate.')
        if not frame.trade_date.is_monotonic_increasing:raise ValueError('Validation dates must be sorted.')
        self.frame=frame[required].reset_index(drop=True).copy()
        self.returns=self.frame.future_oc_return.to_numpy(dtype=float)
        self.ids=self.frame.security_id.to_numpy()
        if pd.isna(self.ids).any():raise ValueError('Unknown security ID.')
        self.known=np.isfinite(self.returns)&self.frame.outcome_status.eq('ok').to_numpy()
        self.groups=[];coverage=[];start=0
        for date,g in self.frame.groupby('trade_date',sort=False):
            end=start+len(g);complete=bool(self.known[start:end].all())
            self.groups.append((start,end,date,complete))
            coverage.append(dict(trade_date=date,forecast_candidates=len(g),unknown_candidate_outcomes=int((~self.known[start:end]).sum()),
                fixed_stopping_date=complete,reason='all_candidate_outcomes_observed' if complete else 'unknown_candidate_outcome_fixed_date_exclusion'))
            start=end
        self.coverage=pd.DataFrame(coverage)
        if int(self.coverage.fixed_stopping_date.sum())<2:raise ValueError('Fewer than two fixed complete-outcome validation dates.')
        self.history=[];self.daily_history=[]

    def evaluate(self,scores,*,record=True):
        scores=np.asarray(scores,dtype=float)
        if scores.shape!=(len(self.frame),) or not np.isfinite(scores).all():raise ValueError('Invalid full-candidate validation scores.')
        rows=[];iteration=len(self.history)+1
        for a,b,date,complete in self.groups:
            s=scores[a:b];r=self.returns[a:b];known=self.known[a:b];ids=self.ids[a:b]
            row=dict(round=iteration,trade_date=date,fixed_stopping_date=complete,forecast_candidates=b-a,
                unknown_candidate_outcomes=int((~known).sum()))
            for n in (2,3):
                if len(s)<2*n:
                    gross=notional=0.;missing=0;status='cash_insufficient_forecasts'
                else:
                    li,si=select_portfolio_indices(s,ids,n)
                    chosen=np.r_[li,si];weights=np.r_[np.full(n,.5/n),np.full(n,-.5/n)]
                    missing=int((~known[chosen]).sum());status='observed' if not missing else 'unresolved_selected_outcome'
                    gross=float(np.sum(weights*r[chosen])) if not missing else np.nan
                    notional=float(np.sum(np.abs(weights)*(2+r[chosen]))) if not missing else np.nan
                row[f'gross_n{n}']=gross;row[f'net_1bp_n{n}']=gross-notional/10000
                row[f'missing_selected_n{n}']=missing;row[f'status_n{n}']=status
            row['mean_gross_n2_n3']=.5*(row['gross_n2']+row['gross_n3'])
            rows.append(row)
        daily=pd.DataFrame(rows)
        fixed=daily.loc[daily.fixed_stopping_date]
        if not np.isfinite(fixed[['gross_n2','gross_n3']]).all().all():raise ValueError('Unknown outcome on a fixed complete-outcome date.')
        result=dict(round=iteration,metric=float(fixed.mean_gross_n2_n3.mean()),
            mean_gross_n2=float(fixed.gross_n2.mean()),mean_gross_n3=float(fixed.gross_n3.mean()),
            mean_net_1bp_n2=float(fixed.net_1bp_n2.mean()),mean_net_1bp_n3=float(fixed.net_1bp_n3.mean()),
            fixed_scoring_dates=len(fixed),all_validation_dates=len(daily),
            excluded_fixed_dates=int((~daily.fixed_stopping_date).sum()),
            unresolved_selected_dates_n2=int(daily.missing_selected_n2.gt(0).sum()),
            unresolved_selected_dates_n3=int(daily.missing_selected_n3.gt(0).sum()))
        if record:self.history.append(result);self.daily_history.extend(rows)
        return result['metric']

    def __call__(self,scores):return self.evaluate(scores)
