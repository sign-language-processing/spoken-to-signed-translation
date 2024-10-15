LANGUAGE_BACKUP = {
    'slf': 'ise',  # Swiss-Italian to Italian
    'ssr': 'fsl'  # Swiss-French to French
}


def languages_set(signed_language: str):
    if signed_language in LANGUAGE_BACKUP:
        return {signed_language}.union(languages_set(LANGUAGE_BACKUP[signed_language]))

    return {signed_language}
