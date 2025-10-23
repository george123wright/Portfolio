
_CCY_BY_SUFFIX = {
    ".L": "GBP",
    ".HK": "HKD",
    ".PA": "EUR",
    ".DE": "EUR", 
    ".MC": "EUR", 
    ".AS": "EUR",  
    ".TO": "CAD",  
    ".CO": "DKK",  
    ".AX": "AUD", 
}


_EU_SUFFIXES = {
    ".L", 
    ".PA", 
    ".DE", 
    ".MC", 
    ".AS", 
    ".CO"
}


_USD_EXCEPTIONS = {
    "EMVL.L", 
    "IWQU.L",
    "IWSZ.L"
}


_REGION_FACTOR_MAP = {
    "US": {
        "VLUE": "VLUE",   
        "QUAL": "QUAL",    
        "MTUM": "MTUM",  
        "USMV": "USMV",  
        "SIZE": "SIZE",    
    },
    "EU": {
        "VLUE": "IEVL.MI",
        "QUAL": "IEQU.MI",
        "MTUM": "IEMO.MI",
        "USMV": "MVEU.MI",
        "SIZE": "IEFS.L",
    },
    "EM": {
        "VLUE": "EMVL.L",
        "QUAL": "IWQU.L",
        "MTUM": "EEMO",
        "USMV": "EEMV",
        "SIZE": "IWSZ.L",
    },
}


_FACTOR_LABEL_TO_CODE = {
    "Value": "VLUE",
    "Quality": "QUAL",
    "Momentum": "MTUM",
    "MinVol": "USMV",
    "Size": "SIZE",
}


_FACTORS = [
    "MTUM", 
    "QUAL",
    "SIZE",
    "USMV", 
    "VLUE"
]


