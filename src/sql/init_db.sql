-- 1. NETTOYAGE (Ordre inverse des dépendances)
DROP TABLE IF EXISTS poi CASCADE;
DROP TABLE IF EXISTS adresse CASCADE;
DROP TABLE IF EXISTS sub_category CASCADE;
DROP TABLE IF EXISTS main_category CASCADE;

-- 2. CRÉATION DES TABLES

-- A. Catégories Principales
CREATE TABLE main_category (
    id INT PRIMARY KEY,
    nom_cat VARCHAR(100) NOT NULL
);

-- B. Sous-Catégories
CREATE TABLE sub_category (
    id INT PRIMARY KEY,
    nom_sous_cat VARCHAR(100) NOT NULL,
    main_category_id INT,
    CONSTRAINT fk_main_cat FOREIGN KEY (main_category_id) REFERENCES main_category(id)
);

-- C. Adresses
CREATE TABLE adresse (
    id SERIAL PRIMARY KEY,
    label_adresse VARCHAR(255),
    rue VARCHAR(255),
    code_postal VARCHAR(20),
    commune VARCHAR(100),
    departement VARCHAR(100),
    region VARCHAR(100)
);

-- D. Points d'Intérêt (POI)
CREATE TABLE poi (
    poi_id SERIAL PRIMARY KEY,
    
    -- Données Métier
    nom_du_poi VARCHAR(255),
    description TEXT,
    
    -- Coordonnées brutes
    latitude FLOAT,
    longitude FLOAT,
    
    -- Clés étrangères
    main_category_id INT,
    sub_category_id INT,
    adresse_id INT,
    
    -- Infos complémentaires
    contact_mail VARCHAR(255),
    contact_phone VARCHAR(255),
    contact_website VARCHAR(255),
    itineraire BOOLEAN,
    
    -- Index Géospatiaux H3
    h3_r6 VARCHAR(15),
    h3_r7 VARCHAR(15),
    h3_r8 VARCHAR(15),
    h3_r9 VARCHAR(15),
    
    -- Scores (Normalisés 0-1)
    density_commune_norm FLOAT DEFAULT 0,
    diversity_commune_norm FLOAT DEFAULT 0,
    popularity_norm FLOAT DEFAULT 0,
    proximity_commune_norm FLOAT DEFAULT 0,
    category_weight_norm FLOAT DEFAULT 0,
    opening_score_norm FLOAT DEFAULT 0,
    final_score FLOAT DEFAULT 0,

    -- 📍 COLONNE GÉOMÉTRIQUE AUTOMATIQUE (PostGIS)
    geom GEOMETRY(Point, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_Point(longitude, latitude), 4326)) STORED,

    -- Contraintes
    CONSTRAINT fk_poi_adresse FOREIGN KEY (adresse_id) REFERENCES adresse(id) ON DELETE SET NULL,
    CONSTRAINT fk_poi_main_cat FOREIGN KEY (main_category_id) REFERENCES main_category(id) ON DELETE SET NULL,
    CONSTRAINT fk_poi_sub_cat FOREIGN KEY (sub_category_id) REFERENCES sub_category(id) ON DELETE SET NULL
);

-- 3. CRÉATION DES INDEX (Performance)
CREATE INDEX idx_pois_geom ON poi USING GIST(geom);
CREATE INDEX idx_poi_main_cat ON poi(main_category_id);
CREATE INDEX idx_poi_sub_cat ON poi(sub_category_id);
CREATE INDEX idx_poi_h3_r9 ON poi(h3_r9);
CREATE INDEX idx_poi_score ON poi(final_score DESC);

-- 4. INSERTION DES DONNÉES DE RÉFÉRENCE (Généré automatiquement)

-- Insertion Main Categories
INSERT INTO main_category (id, nom_cat) VALUES 
(0, 'Nature & Paysages'),
(1, 'Information Touristique'),
(2, 'Bien-être & Santé'),
(3, 'Famille & Enfants'),
(4, 'Transports'),
(5, 'Commodités'),
(6, 'Événements & Traditions'),
(7, 'Commerce & Shopping'),
(8, 'Gastronomie & Restauration'),
(9, 'Shopping & Artisanat'),
(10, 'Culture & Musées'),
(11, 'Santé & Urgences'),
(12, 'Hébergement'),
(13, 'Sports & Loisirs'),
(14, 'Services & Mobilité'),
(15, 'Transports touristiques'),
(16, 'Loisirs & Clubs'),
(17, 'Camping & Plein Air'),
(18, 'Patrimoine & Monuments');

-- Insertion Sub Categories
INSERT INTO sub_category (id, nom_sous_cat, main_category_id) VALUES 
(0, 'Restauration rapide', 8),
(1, 'Châteaux & Fortifications', 18),
(2, 'Religieux', 18),
(3, 'Côtes & littoral', 0),
(4, 'Sports de balle & raquette', 13),
(5, 'Artisanat', 9),
(6, 'Eau & Milieux humides', 0),
(7, 'Commerces', 9),
(8, 'Bibliothèques & médiation', 10),
(9, 'Loisirs indoor', 13),
(10, 'Producteurs', 8),
(11, 'unknown', 7),
(12, 'Antiquités & brocante', 9),
(13, 'Restaurants', 8),
(14, 'Marchés', 9),
(15, 'Parcs & loisirs', 3),
(16, 'unknown', 12),
(17, 'Éducation & apprentissage', 3),
(18, 'Forêts & milieux naturels', 0),
(19, 'Aire de pique-nique', 6),
(20, 'Sports équestres', 13),
(21, 'Religieux', 6),
(22, 'Trains & bus touristiques', 15),
(23, 'unknown', 4),
(24, 'Zoo & animaux', 3),
(25, 'Spectacle vivant', 10),
(26, 'Bars & cafés', 8),
(27, 'unknown', 1),
(28, 'Rencontres & conférences', 10),
(29, 'Sports nautiques', 13),
(30, 'Paysages remarquables', 0),
(31, 'Montagne & Relief', 0),
(32, 'unknown', 16),
(33, 'Ouvrages d''art', 18),
(34, 'Produits locaux', 8),
(35, 'Antiquité & Vestiges', 18),
(36, 'Golf & mini-golf', 13),
(37, 'Musées & expositions', 10),
(38, 'Téléphériques & remontées', 15),
(39, 'Eau vive & cascades', 0),
(40, 'Aires & jeux', 3),
(41, 'Patrimoine rural & agricole', 18),
(42, 'Thermalisme', 2),
(43, 'Sports collectifs & stades', 13),
(44, 'Cinéma & audiovisuel', 10),
(45, 'unknown', 14),
(46, 'Jeune public', 10),
(47, 'Géologie & curiosités', 0),
(48, 'Sports mécaniques', 13),
(49, 'Patrimoine civil', 18),
(50, 'Sports outdoor', 13),
(51, 'Concerts & musique', 10),
(52, 'unknown', 17),
(53, 'Fêtes & traditions', 6),
(54, 'Festivals & grands événements', 10),
(55, 'Soins & bien-être', 2),
(56, 'unknown', 5),
(57, 'Foires & salons', 9),
(58, 'Cimetières & mémoriaux', 18),
(59, 'unknown', 11),
(60, 'Glace & haute montagne', 0),
(61, 'Sports d''hiver', 13),
(62, 'Thalasso & balnéo', 2),
(63, 'Aventure & accrobranche', 13),
(64, 'Défilés & parades', 6),
(65, 'Vins & spiritueux', 8);

-- Fin du script
