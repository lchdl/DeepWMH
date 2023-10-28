class TableView:
    def __init__(self, rows: int, cols: int, title : str = '', style:str = 'triple', column_width=[]):
        assert style in self._valid_table_styles(), 'Invalid table style. Must be one of the following: %s.' % (', '.join(self._valid_table_styles()))
        assert rows > 0 and cols > 0, 'Invalid settings for row and column size.'
        assert len(column_width) <= cols, 'Invalid width setting.'
        self.rows = rows
        self.cols = cols
        self.table = [['' for j in range(cols)] for i in range(rows)]
        self.style = style
        self.title = title
        self.column_size = [0] * self.cols
        self.column_max_width = [0] * self.cols
        for t in range(len(column_width)):
            self.column_max_width[t] = column_width[t]
    
    def set(self,row_id: int, col_id: int, content: str):
        assert isinstance(row_id, int)
        assert isinstance(col_id, int)
        assert row_id >= 0 and row_id < self.rows, 'invalid row number. (%d~%d, got %d).' % (0, self.rows-1, row_id)
        assert col_id >= 0 and col_id < self.cols, 'invalid column number. (%d~%d, got %d).' % (0, self.cols-1, col_id)
        assert isinstance(content, str), 'Internal error, content must be string. Position: (%d, %d). Content: "%s".' % (row_id, col_id, content)

        self.table[row_id][col_id] = content

    def show(self):
        self._calc_column_size()
        print('')
        if self.style == 'triple': self._show_triple()
        print('')

    def _calc_column_size(self):
        for col in range(self.cols):
            l = 0
            for row in range(self.rows):
                t = len(self.table[row][col])
                if l < t: l = t
            self.column_size[col] = l
                
    def _valid_table_styles(self):
        return ['triple']

    def _lsum(self,l):
        sum = 0
        for item in l:
            sum += item
        return sum
        
    def _show_triple(self):

        # calculate full width
        full_width = 0
        for i in range(self.cols):
            if self.column_max_width[i] > 0:
                full_width += self.column_max_width[i]
            else:
                full_width += self.column_size[i]
        full_width += self.cols + 1

        d = '=' * full_width
        s = '-' * full_width

        # print title
        if len(self.title) > 0:
            pad = (len(d) - len(self.title)) // 2
            print(' '*pad + self.title)

        print(d)
        for i in range(self.cols):
            item = self.table[0][i]
            item = item + ' '* (self.column_size[i]-len(item))
            if self.column_max_width[i] != 0:
                if len(item) > self.column_max_width[i]:
                    item = item[:self.column_max_width[i]]
            print(' ' + item, end='')
        print('')
        print(s)
        for row in range(1, self.rows):
            for col in range(self.cols):
                item = self.table[row][col]
                item = item + ' '* (self.column_size[col]-len(item))
                if self.column_max_width[col] != 0:
                    if len(item) > self.column_max_width[col]:
                        item = item[:self.column_max_width[col]]
                print(' ' + item, end='')
            print('')
        print(d)

