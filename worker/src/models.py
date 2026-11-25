from redbox.models.file import FileBase


class File(FileBase):
    class Meta:
        app_label = "redbox_core"
        db_table = "redbox_core_file"
        managed = False
