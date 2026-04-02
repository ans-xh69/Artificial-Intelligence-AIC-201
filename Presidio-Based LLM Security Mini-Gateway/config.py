LOW_THRESHOLD = 3
HIGH_THRESHOLD = 6

PATTERNS = {
    "instruction_override": { "keywords": [
            "ignore instructions",
            "ignore previous instructions",
            "ignore all previous instructions",
            "forget rules",
            "forget your policies",
            "disregard your safety policies"
        ], "weight": 3
    },
    
    "jailbreak": { "keywords": [
            "developer mode",
            "unrestricted mode",
            "no restrictions",
            "disable safeguards",
            "answer every question honestly",
            "remove all safety restrictions"
        ], "weight": 3
    },
    
    "prompt_extraction": { "keywords": [
            "system prompt",
            "hidden instructions",
            "reveal prompt",
            "display internal prompt"
        ], "weight": 4
    },
    
    "data_exfiltration": { "keywords": [
            "list emails",
            "api key",
            "database",
            "password",
            "provide the list"
        ], "weight": 4
    },
    
    "bypass": { "keywords": [
            "bypass",
            "disable safeguards",
            "remove all safety restrictions"
        ], "weight": 2
    }
}