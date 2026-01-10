import pandas as pd
# définition des variables globales main_types/main_types_linked contenant repectivement les 4 types principaux de POI
# sur la plateforme DataToursime et les labels correspondants utilisés dans les données :

main_types = ['Place', 'Route', 'Product', 'Entertainment and event']
main_types_linked = ['PlaceOfInterest', 'Tour', 'Product', 'EntertainmentAndEvent']

main_types_fr = ['Itinéraire touristique', 'Fête et manifestation', 'Lieu', 'Produit']  # intitulés des 4 types principaux de POI en français

# nouvelles types à ajouter au mapping des types - catégories des POIs qui sont issus de l'analyse des POI
new_categories_mapping_df = pd.DataFrame({'to_keep' : [1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2],
                                           'main_category': ['Shopping & Artisanat', 'Famille & Enfants', 'Famille & Enfants', 'Sports & Loisirs',
                                                              'Gastronomie & Restauration', 'Hébergement','Culture & Musées', 'Culture & Musées',
                                                               'Hébergement', 'Hébergement', 'Culture & Musées',
                                                              'Nature & Paysages', 'Gastronomie & Restauration', 'Famille & Enfants', 
                                                              'Famille & Enfants', 'Hébergement' ] ,
                                            'sub_category' : ['Commerces', 'Parcs & loisirs', 'Parcs & loisirs', 'Sports collectifs & stades',
                                                               'Vins & spiritueux', 'Unknown',
                                                                'Musées & expositions', 'Concerts & musique', 'Unknown', 'Unknown', 'Cinéma & audiovisuel',
                                                                'Forêts & milieux naturels', 'Bars & cafés', 'Zoo & animaux', 'Famille & Enfants', 'Unknown' ],
                                             'linked_label' : ['LocalBusiness', 'AmusementPark', 'Park', 'StadiumOrArena', 'Winery', 'LodgingBusiness',
                                                                'ExhibitionEvent', 'MusicEvent','BedAndBreakfast', 'TableHoteGuesthouse', 'MovieTheater',
                                                                'Landform', 'CafeOrCoffeeShop', 'Aquarium', 'Zoo', 'Hostel']})

