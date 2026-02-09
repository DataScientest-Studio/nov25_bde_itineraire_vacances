class CategoryRepository:

    def get_main_categories(self, db):
        with db.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT(nom_cat)
                FROM main_category AS m
                LEFT JOIN sub_category AS s
                ON m.id = s.main_category_id
                WHERE nom_sous_cat <> 'unknown';;
            """)
            rows = cur.fetchall()
            main_categories_list = [row[0] for row in rows]
        return main_categories_list

    def get_sub_categories(self, db, main_categories: list[str]):
        query = """SELECT DISTINCT(nom_sous_cat)
                FROM sub_category
                WHERE main_category_id IN (SELECT id FROM main_category WHERE nom_cat IN %s )
                ORDER BY nom_sous_cat;
            """
        with db.cursor() as cur:
            cur.execute(query, (tuple(main_categories),))
            rows = cur.fetchall()

        return [row[0] for row in rows if row[0] is not None]



category_repository = CategoryRepository()