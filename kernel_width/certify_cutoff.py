"""Outward-rounded checks of the elementary margins in Proposition cutoff.

This certifies scalar inequalities only. The note supplies the geometric
argument turning them into the exact energy difference -2*m**2.
"""
from pathlib import Path
import json
from mpmath import iv

iv.dps = 50
r, d = iv.mpf(499)/1000, iv.mpf(501)/1000
cutoff = iv.mpf(19)/20
cos_limit = iv.cos(iv.pi/24)
margins = {
    'straight_cos_a_minus_cos_pi24': r/d-cos_limit,
    'curved_cos_a_minus_cos_pi24': 1/(2*d*d)-1-cos_limit,
    'cutoff_squared_minus_upper_distance_squared':
        cutoff**2-((d-r)**2+d*d*(2+iv.sqrt(2))),
    'lower_distance_squared_minus_cutoff_squared':
        (2*r*iv.cos(iv.pi/16))**2-cutoff**2,
}
assert all(v.a > 0 for v in margins.values())
result = {
    'method': 'mpmath.iv, 50 decimal digits, outward-rounded interval arithmetic',
    'margins': {k: str(v) for k, v in margins.items()},
    'conclusion': 'All scalar margins are strictly positive; energy difference is -2*m^2.'
}
Path(__file__).with_name('cutoff_certificate.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
