class POIRepository:

    def filter_pois(self, db, filters, h3_r7):

        query = """
            SELECT p.poi_id,
                p.nom_du_poi,
                p.description,
                p.longitude,
                p.latitude,
                CONCAT_WS(
                    ', ',
                    TRIM(BOTH '[]''' FROM ad.rue),
                    CONCAT(ad.code_postal, ' ', ad.commune),
                    ad.departement,
                    ad.region
                ) AS adresse,
                mc.nom_cat AS main_category,
                sc.nom_sous_cat AS sub_category,
                p.contact_mail,
                p.contact_phone,
                p.contact_website,
                p.itineraire,
                p.h3_r7,
                p.diversity_commune_norm,
                p.final_score
            FROM poi AS p
            JOIN adresse AS ad
                ON p.adresse_id = ad.id
            JOIN main_category AS mc
                ON p.main_category_id = mc.id
            JOIN sub_category AS sc
                ON p.sub_category_id = sc.id
            WHERE p.h3_r7 = ANY(%s)  -- pré-filtrage H3
            AND ST_DWithin(
                    p.geom,
                    ST_SetSRID(ST_Point(%s, %s), 4326),
                    %s
                )
            AND mc.nom_cat = ANY(%s)
            AND sc.nom_sous_cat = ANY(%s);
            """

        with db.cursor() as cur:
            cur.execute(
                query,
                (
                    h3_r7,                   # %s 1
                    filters.longitude,       # %s 2
                    filters.latitude,        # %s 3
                    filters.radius,          # %s 4
                    filters.main_category,   # %s 5
                    filters.sub_category     # %s 6
                )
            )
            rows = cur.fetchall()

        # Récupérer les noms de colonnes
        columns = [desc[0] for desc in cur.description]
        

        # Transformer en liste de dicts
        return [dict(zip(columns, row)) for row in rows]


poi_repository = POIRepository()

