from cox.store import Store, _clean_dict


class CustomStore(Store):

    def log_tb(self, table_name, update_dict, summary_type='scalar', step=None, **kwargs):
        """
        Log to only tensorboard.

        Args:
            table_name (str) : which table to log to
            update_dict (dict) : values to log and store as a dictionary of
                column mapping to value.
            summary_type (str) : what type of summary to log to tensorboard as
        """

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)

        tb_func = getattr(self.tensorboard, 'add_%s' % summary_type)
        step = step if step else table.nrows

        for name, value in update_dict.items():
            tb_func('/'.join([table_name, name]), value, step, **kwargs)

        return update_dict

    def log_table_and_tb(self, table_name, update_dict, summary_type='scalar', step=None, **kwargs):
        """
        Log to a table and also a tensorboard.

        Args:
            table_name (str) : which table to log to
            update_dict (dict) : values to log and store as a dictionary of
                column mapping to value.
            summary_type (str) : what type of summary to log to tensorboard as
        """

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)

        if hasattr(self, 'tensorboard'):
            tb_func = getattr(self.tensorboard, 'add_%s' % summary_type)
            step = step if step else table.nrows

            for name, value in update_dict.items():
                tb_func('/'.join([table_name, name]), value, step, **kwargs)

        table.update_row(update_dict)

    def update_row(self, table_name, update_dict):

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)
        table.update_row(update_dict)

    def load(self, table, key, data_save_type, iteration=-1, **kwargs):

        if data_save_type == 'object':
            return self.tables[table].get_object(self.tables[table].df[key].iloc[iteration], **kwargs)
        elif data_save_type == 'state_dict':
            return self.tables[table].get_state_dict(self.tables[table].df[key].iloc[iteration], **kwargs)
        elif data_save_type == 'pickle':
            return self.tables[table].get_pickle(self.tables[table].df[key].iloc[iteration], **kwargs)
        else:
            return self.tables[table].df[key].iloc[iteration]
