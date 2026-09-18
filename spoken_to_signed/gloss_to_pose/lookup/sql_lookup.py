"""Optional PostgreSQL captions backend, migrated from sign/models."""

from contextlib import closing

from spoken_to_signed.gloss_to_pose.languages import languages_set
from spoken_to_signed.text_to_gloss.types import Gloss, GlossItem

from .lookup import PoseLookup, gloss_candidates


class SQLPoseLookup(PoseLookup):
    def __init__(self, database_config: dict, backup: PoseLookup = None, pose_prefix="gs://sign-mt-poses"):
        super().__init__(rows=[], backup=backup)
        self.database_config = database_config
        self.pose_prefix = pose_prefix.rstrip("/")

    def query(self, sql: str, params):
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # One connection per batch; never share a connection/cursor across requests.
        with closing(psycopg2.connect(**self.database_config)) as connection:
            connection.set_session(readonly=True, autocommit=True)
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()

    def get_initial_candidates(self, glosses: Gloss, spoken_language: str, signed_language: str, source=None):
        # Match exactly the forms PoseLookup will try, including atomic multiword items.
        terms = sorted({term.lower() for word, gloss in glosses for term in [word or gloss, *gloss_candidates(gloss)]})
        rows = self.query(
            """
            SELECT * FROM (
                SELECT "videoId", language, "videoLanguage", start, "end",
                       unnest(string_to_array(text, ' / ')) AS phrase,
                       unnest(string_to_array(lemmas, ' / ')) AS gloss
                FROM captions
                WHERE language = %s AND "videoLanguage" = ANY(%s)
                      AND start = 0 AND strpos("videoId", %s) = 1
            ) AS candidates
            WHERE phrase IS NOT NULL AND (lower(phrase) = ANY(%s) OR lower(gloss) = ANY(%s))
            ORDER BY ("end" - start) DESC, "videoId", phrase
            """,
            (spoken_language, sorted(languages_set(signed_language)), source or "", terms, terms),
        )
        return [
            {
                "path": f"{self.pose_prefix}/{row['videoId']}.pose",
                "spoken_language": row["language"],
                "signed_language": row["videoLanguage"],
                "start": row["start"],
                "end": row["end"],
                "words": row["phrase"],
                "glosses": row["gloss"] or "",
                "priority": 0,
            }
            for row in rows
        ]

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source=None):
        return self.lookup_sequence([GlossItem(word, gloss)], spoken_language, signed_language, source)[0]

    def lookup_sequence(self, glosses: Gloss, spoken_language: str, signed_language: str, source=None):
        if not glosses:
            self.last_coverage = []
            return []
        glosses = [GlossItem(word or gloss, gloss) for word, gloss in glosses]
        rows = self.get_initial_candidates(glosses, spoken_language, signed_language, source)
        lookup = PoseLookup(rows=rows, backup=self.backup, cache=self.cache)
        results = lookup.lookup_sequence(glosses, spoken_language, signed_language)
        self.last_coverage = lookup.last_coverage
        return results
