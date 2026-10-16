"""Add the evidenced TSS/GPN stock entitlement; never invent retired quotes.

Closing filings distinguish legal September17 consummation from September18
first suspended session and announcement. The engine applies the mandatory
entitlement before September18 open, retaining theoretical fractional units
under its existing proxy instead of inventing broker cash-in-lieu amounts.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .action_loading import EVENT_TERMS
from .sni_action import SNI_EVENT,_verify_pin
from .universe import DataContractError

TSS_PREDECESSOR_ID='sec_XLD6OX'
TSS_SUCCESSOR_ID='sec_z95Pqz'
TSS_EVENT={
 'event_id':'TSS-2019-09-17','predecessor_id':TSS_PREDECESSOR_ID,
 'effective_date':'2019-09-18','legal_effective_date':'2019-09-17',
 'announcement_date':'2019-09-18','successor_id':TSS_SUCCESSOR_ID,
 'cash_per_raw_share':0.,'stock_ratio':.8101,'verified':True,
 'evidence_reference':'https://www.sec.gov/Archives/edgar/data/721683/000119312519250774/d803104d8k.htm',
 'timing_evidence_reference':'https://www.sec.gov/Archives/edgar/data/721683/000119312519250774/d803104d8k.htm',
 'corroborating_evidence_reference':'https://www.sec.gov/Archives/edgar/data/1123360/000119312519250768/d801793d8k.htm',
 'settlement_verified':False,'knowledge_time_verified':False,
 'successor_ticker':'GPN','successor_share_class':'Global Payments common stock, no par value',
 'successor_isin':'US37940X1028',
 'successor_identity_status':'provisional_provider_mapping_supported_by_exact_common_share_class_and_isin',
 'suspension_timing':'before_NYSE_open_on_application_session',
 'ordinary_cash_consideration':0.,
 'fractional_share_policy':'theoretical_fractional_units_retained_no_cash_in_lieu_invented',
 'actual_fractional_share_terms':'issuer_pays_cash_in_lieu_actual_amount_and_timing_unverified',
 'new_order_after_retirement':'unfilled_cash_no_merger_entitlement',
}

def validate_tss_action_prices(prices):
 if not isinstance(prices,pd.DataFrame):raise DataContractError('TSS action requires observed prices')
 date_field='trading_date' if 'trading_date' in prices else 'date'
 required={date_field,'intrinio_security_id','open','close','adj_open','adj_close','frequency','intraperiod','price_quality_status'}
 if not required<=set(prices):raise DataContractError('TSS conversion requires original paired raw/adjusted price fields')
 try:dates=pd.DatetimeIndex(pd.to_datetime(prices[date_field],errors='raise'))
 except (ValueError,TypeError,OverflowError) as exc:raise DataContractError('Malformed TSS action quote date') from exc
 if (dates.tz is not None or dates.isna().any() or not dates.equals(dates.normalize()) or dates.min()<pd.Timestamp('2018-01-01') or dates.max()>pd.Timestamp('2019-12-31')):raise DataContractError('TSS financial evidence must remain within2018–2019')
 for native,day,fields in ((TSS_PREDECESSOR_ID,'2019-09-17',('close','adj_close')),(TSS_SUCCESSOR_ID,'2019-09-18',('open','adj_open'))):
  selected=prices.loc[prices.intrinio_security_id.eq(native).to_numpy()&(dates==pd.Timestamp(day))]
  if len(selected)!=1:raise DataContractError('TSS action lacks a unique observed predecessor/successor basis')
  row=selected.iloc[0]
  if row.frequency!='daily' or not isinstance(row.intraperiod,(bool,np.bool_)) or bool(row.intraperiod) or row.price_quality_status!='accepted':raise DataContractError('TSS conversion quote must be an accepted completed daily observation')
  for field in fields:
   value=row[field]
   if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<=0:raise DataContractError('TSS conversion requires positive finite paired quote units')
 if (prices.intrinio_security_id.eq(TSS_PREDECESSOR_ID).to_numpy()&(dates>=pd.Timestamp('2019-09-18'))).any():raise DataContractError('Cannot quote retired TSS as tradable after verified suspension')

def load_tss_action(evidence_directory):
 directory=Path(evidence_directory).resolve()
 with (directory/'tss_action_evidence.json').open() as stream:receipt=json.load(stream)
 if (receipt.get('schema')!='dmv2-tss-action-evidence/v1' or receipt.get('event')!=TSS_EVENT or receipt.get('no_new_quote_created') is not True or receipt.get('successor_class_verified') is not True or receipt.get('settlement_or_fractional_cash_invented') is not False):raise DataContractError('TSS event evidence/terms changed')
 artifacts=receipt.get('artifacts',{})
 if set(artifacts)!={'successor_identity.json','predecessor_successor_controls.parquet'}:raise DataContractError('TSS evidence omits exact bounded identity/quote controls')
 paths={name:_verify_pin(directory,pin) for name,pin in artifacts.items()}
 with paths['successor_identity.json'].open() as stream:identity=json.load(stream)
 master=identity.get('master',{});links=identity.get('links',[])
 if (master.get('intrinio_security_id')!=TSS_SUCCESSOR_ID or master.get('ticker')!='GPN' or master.get('type')!='Ordinary Shares' or not any(link.get('intrinio_security_id')==TSS_SUCCESSOR_ID and link.get('query_identifier_type')=='isin' and link.get('query_identifier')=='US37940X1028' and link.get('resolution_status')=='resolved' for link in links)):raise DataContractError('TSS successor must be Global Payments common shares with reviewed ISIN route')
 validate_tss_action_prices(pd.read_parquet(paths['predecessor_successor_controls.parquet']))
 return deepcopy(TSS_EVENT)

def append_tss_action(events,evidence_directory,*,prices=None):
 """Preserve all ten existing dictionaries and append the eleventh action."""
 from .vvc_action import VVC_EVENT
 original=events.to_dict('records') if isinstance(events,pd.DataFrame) else list(events)
 expected={ticker+'-'+terms[1] for ticker,terms in EVENT_TERMS.items()}|{SNI_EVENT['event_id'],VVC_EVENT['event_id']}
 if len(original)!=10 or not all(isinstance(row,dict) for row in original) or {row.get('event_id') for row in original}!=expected:raise DataContractError('TSS amendment requires all unchanged original ten actions')
 for row in original:
  if row['event_id']==SNI_EVENT['event_id']:
   if row!=SNI_EVENT:raise DataContractError('TSS amendment cannot alter SNI terms')
  elif row['event_id']==VVC_EVENT['event_id']:
   if row!=VVC_EVENT:raise DataContractError('TSS amendment cannot alter VVC terms')
  else:
   terms=EVENT_TERMS[row['event_id'].split('-',1)[0]]
   if (row.get('predecessor_id'),row.get('effective_date'),row.get('cash_per_raw_share'),row.get('stock_ratio'))!=(terms[0],terms[2],terms[3],terms[4]):raise DataContractError('TSS amendment cannot alter original eight event terms')
 action=load_tss_action(evidence_directory)
 if prices is not None:validate_tss_action_prices(prices)
 return deepcopy(original)+[action]
