from app.repositories.category_repository import category_repository

class CategoryService:

    def get_main_categories(self, db):
        return category_repository.get_main_categories(db)

    
    def get_sub_categories(self, db, main_categories: list[str]):
        return category_repository.get_sub_categories(db, main_categories)


category_service = CategoryService()