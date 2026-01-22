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
    source_uri VARCHAR(255) UNIQUE,
    nom_du_poi VARCHAR(255) NOT NULL,
    description TEXT,
    latitude FLOAT,
    longitude FLOAT,
    
    -- Clés étrangères
    adresse_id INT,
    main_category_id INT,
    sub_category_id INT,
    
    -- Infos
    site_web VARCHAR(500),
    telephone VARCHAR(50),
    email VARCHAR(150),
    contacts_json TEXT,
    itineraire TEXT, -- Ajouté pour stocker le JSON itinéraire
    
    -- H3 Index
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

    -- Contraintes
    CONSTRAINT fk_poi_adresse FOREIGN KEY (adresse_id) REFERENCES adresse(id) ON DELETE SET NULL,
    CONSTRAINT fk_poi_main_cat FOREIGN KEY (main_category_id) REFERENCES main_category(id) ON DELETE SET NULL,
    CONSTRAINT fk_poi_sub_cat FOREIGN KEY (sub_category_id) REFERENCES sub_category(id) ON DELETE SET NULL
);

-- E. Index de performance
CREATE INDEX idx_poi_main_cat ON poi(main_category_id);
CREATE INDEX idx_poi_sub_cat ON poi(sub_category_id);
CREATE INDEX idx_poi_h3_r9 ON poi(h3_r9);
CREATE INDEX idx_poi_score ON poi(final_score DESC);

-- 3. INSERTION DES DONNÉES DE RÉFÉRENCE (CATÉGORIES)

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
-- Note : Les apostrophes (ex: d'art) sont doublées (d''art) pour le SQL
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
(37, 'Musées & expositions', 10);
