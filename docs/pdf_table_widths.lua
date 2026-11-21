-- Keep the four-column experiment inventory inside the PDF text block.
function Table(table)
  if #table.colspecs == 4 then
    table.colspecs = {
      { pandoc.AlignLeft, 0.17 },
      { pandoc.AlignLeft, 0.27 },
      { pandoc.AlignLeft, 0.23 },
      { pandoc.AlignLeft, 0.33 },
    }
  end
  return table
end
