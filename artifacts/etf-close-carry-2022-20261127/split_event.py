"""Apply the published terms of the one 2022 share split in an isolated ledger replay."""
from datetime import date
from decimal import Decimal, ROUND_CEILING

import pandas as pd

SYMBOL = "510500"
RECORD = date(2022,8,26)
EX_DATE = date(2022,8,29)
EXACT = Decimal("1.14539")
ARCHIVE = Decimal("1.1454")
SOURCE_URL = "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-08-23/510500_20220823_1_aMMa3ZNB.pdf"


def corrected_actions(actions):
    result=actions.copy()
    on_date=pd.to_datetime(result["ex_date"]).dt.date.eq(EX_DATE)
    event=on_date & result["symbol"].eq(SYMBOL)
    if not event.any():
        return result
    row,=result.loc[event].itertuples(index=False)
    if (pd.Timestamp(row.record_date).date()!=RECORD or row.cash_per_share!=0 or
            Decimal(str(row.share_multiplier)) not in (ARCHIVE,EXACT)):
        raise ValueError("The archived record does not match the published split event")
    # The ledger's internal method receives a date and ratio, but no symbol. Require uniqueness.
    available=pd.to_datetime(result["ex_date"])
    if "share_available_date" in result:
        available=pd.to_datetime(result["share_available_date"]).fillna(available)
    conflicting=~event & available.dt.date.eq(EX_DATE) & result["share_multiplier"].eq(float(EXACT))
    if conflicting.any() or available.loc[event].iloc[0].date()!=EX_DATE:
        raise ValueError("Ambiguous or unexpected settlement date for the 2022 ETF split")
    result.loc[event,"share_multiplier"]=float(EXACT)
    return result


def official_units(registered_units):
    if registered_units<0 or int(registered_units)!=registered_units:
        raise ValueError("Invalid number of registered units")
    return int((Decimal(int(registered_units))*EXACT).to_integral_value(rounding=ROUND_CEILING))


def configure(ledger_module, actions, *, snapshot_mode=False):
    """Install the narrow event override only in the calling replay worker."""
    position=ledger_module._Position
    current=position.apply_split
    original=getattr(current,"_official_split_original",current)
    if snapshot_mode:
        position.apply_split=original
        return actions.copy()
    corrected=corrected_actions(actions)

    def apply_split(self,multiplier,available,*,truncate_fractional=False):
        if available==EX_DATE and truncate_fractional and Decimal(str(multiplier))==EXACT:
            # Registered holder is one account. Round its aggregate once, preserving old lots.
            extra=official_units(self.quantity)-self.quantity
            if extra:
                self.lots.append(ledger_module._Lot(extra,available))
            return
        return original(self,multiplier,available,truncate_fractional=truncate_fractional)

    apply_split._official_split_original=original
    position.apply_split=apply_split
    return corrected
