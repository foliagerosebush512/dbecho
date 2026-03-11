from dbecho.server import _format_table, _format_size


class TestFormatTable:
    def test_no_columns(self):
        assert _format_table([], []) == "(no columns)"

    def test_no_rows(self):
        assert _format_table(["a", "b"], []) == "(no rows)"

    def test_basic(self):
        result = _format_table(["name", "age"], [["Alice", 30], ["Bob", 25]])
        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "30" in result

    def test_none_values(self):
        result = _format_table(["a"], [[None]])
        assert "NULL" in result

    def test_short_row_padded(self):
        result = _format_table(["a", "b", "c"], [[1]])
        assert "NULL" in result

    def test_long_values_truncated(self):
        long_val = "x" * 100
        result = _format_table(["col"], [[long_val]])
        lines = result.split("\n")
        for line in lines:
            assert len(line) <= 62  # 60 + " |" padding

    def test_separator_alignment(self):
        result = _format_table(["name", "value"], [["a", "b"]])
        lines = result.split("\n")
        assert len(lines) == 3  # header, separator, 1 row
        assert "+-" in lines[1]


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(0) == "0 B"
        assert _format_size(512) == "512 B"

    def test_kilobytes(self):
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1536) == "1.5 KB"

    def test_megabytes(self):
        assert _format_size(1048576) == "1.0 MB"

    def test_gigabytes(self):
        assert _format_size(1073741824) == "1.0 GB"

    def test_terabytes(self):
        assert _format_size(1099511627776) == "1.0 TB"

    def test_float_input(self):
        assert _format_size(1024.0) == "1.0 KB"
