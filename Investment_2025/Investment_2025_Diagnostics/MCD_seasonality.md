---
author:
- Lajos Galambos
title: Seasonality
toc-title: Table of contents
---

## McDonald's (MCD) seasonality analysis

I hold MCD stock in my portfolio. Recently, it has made significant
gains, and I am interested in understanding the seasonality of its
returns. I will analyze the average monthly returns of MCD stock over
the past 10 years to identify any seasonal patterns.

I take data from Yahoo Finance for McDonald's (MCD) stock from March 10,
2000, to March 10, 2025. I calculate the monthly returns and group them
by month to compute the average monthly return for each month.

::: cell
::: cell-output-display
![](MCD_seasonality_files/figure-markdown/unnamed-chunk-2-1.png)
:::
:::

I also create a heatmap to visualize the seasonality of MCD stock over
the past 10 years.

::: cell
::: cell-output-display
![](MCD_seasonality_files/figure-markdown/unnamed-chunk-3-1.png)
:::
:::

Finally, lets count the number of positive and negative monthly returns
for each month to see if there is a pattern in the number of positive
and negative months for each month.

::: cell
::: cell-output-display
```{=html}
<div id="jqrqunutdb" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#jqrqunutdb table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#jqrqunutdb thead, #jqrqunutdb tbody, #jqrqunutdb tfoot, #jqrqunutdb tr, #jqrqunutdb td, #jqrqunutdb th {
  border-style: none;
}

#jqrqunutdb p {
  margin: 0;
  padding: 0;
}

#jqrqunutdb .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 14px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#jqrqunutdb .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#jqrqunutdb .gt_title {
  color: #333333;
  font-size: 18px;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#jqrqunutdb .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#jqrqunutdb .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#jqrqunutdb .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#jqrqunutdb .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#jqrqunutdb .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: bold;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#jqrqunutdb .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: bold;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#jqrqunutdb .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#jqrqunutdb .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#jqrqunutdb .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#jqrqunutdb .gt_spanner_row {
  border-bottom-style: hidden;
}

#jqrqunutdb .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#jqrqunutdb .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#jqrqunutdb .gt_from_md > :first-child {
  margin-top: 0;
}

#jqrqunutdb .gt_from_md > :last-child {
  margin-bottom: 0;
}

#jqrqunutdb .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#jqrqunutdb .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#jqrqunutdb .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#jqrqunutdb .gt_row_group_first td {
  border-top-width: 2px;
}

#jqrqunutdb .gt_row_group_first th {
  border-top-width: 2px;
}

#jqrqunutdb .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#jqrqunutdb .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#jqrqunutdb .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#jqrqunutdb .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#jqrqunutdb .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#jqrqunutdb .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#jqrqunutdb .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#jqrqunutdb .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#jqrqunutdb .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#jqrqunutdb .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#jqrqunutdb .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#jqrqunutdb .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#jqrqunutdb .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#jqrqunutdb .gt_left {
  text-align: left;
}

#jqrqunutdb .gt_center {
  text-align: center;
}

#jqrqunutdb .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#jqrqunutdb .gt_font_normal {
  font-weight: normal;
}

#jqrqunutdb .gt_font_bold {
  font-weight: bold;
}

#jqrqunutdb .gt_font_italic {
  font-style: italic;
}

#jqrqunutdb .gt_super {
  font-size: 65%;
}

#jqrqunutdb .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#jqrqunutdb .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#jqrqunutdb .gt_indent_1 {
  text-indent: 5px;
}

#jqrqunutdb .gt_indent_2 {
  text-indent: 10px;
}

#jqrqunutdb .gt_indent_3 {
  text-indent: 15px;
}

#jqrqunutdb .gt_indent_4 {
  text-indent: 20px;
}

#jqrqunutdb .gt_indent_5 {
  text-indent: 25px;
}

#jqrqunutdb .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}

#jqrqunutdb div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_heading">
      <td colspan="4" class="gt_heading gt_title gt_font_normal gt_bottom_border" style>Historical Monthly Returns: Positive vs. Negative</td>
    </tr>
    
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Negative">Negative Months</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Positive">Positive Months</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Total">Total Years</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Positive_Percentage">Positive %</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Jan">Jan</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Jan  Negative" class="gt_row gt_right">13</td>
<td headers="Jan  Positive" class="gt_row gt_right">12</td>
<td headers="Jan  Total" class="gt_row gt_right">25</td>
<td headers="Jan  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFA100; color: #000000;">48%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Feb">Feb</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Feb  Negative" class="gt_row gt_right">11</td>
<td headers="Feb  Positive" class="gt_row gt_right">14</td>
<td headers="Feb  Total" class="gt_row gt_right">25</td>
<td headers="Feb  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFD100; color: #000000;">56%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Mar">Mar</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Mar  Negative" class="gt_row gt_right">8</td>
<td headers="Mar  Positive" class="gt_row gt_right">18</td>
<td headers="Mar  Total" class="gt_row gt_right">26</td>
<td headers="Mar  Positive_Percentage" class="gt_row gt_right" style="background-color: #E7FF00; color: #000000;">69%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Apr">Apr</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Apr  Negative" class="gt_row gt_right">6</td>
<td headers="Apr  Positive" class="gt_row gt_right">19</td>
<td headers="Apr  Total" class="gt_row gt_right">25</td>
<td headers="Apr  Positive_Percentage" class="gt_row gt_right" style="background-color: #C4FF00; color: #000000;">76%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="May">May</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="May  Negative" class="gt_row gt_right">12</td>
<td headers="May  Positive" class="gt_row gt_right">13</td>
<td headers="May  Total" class="gt_row gt_right">25</td>
<td headers="May  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFBA00; color: #000000;">52%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Jun">Jun</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Jun  Negative" class="gt_row gt_right">17</td>
<td headers="Jun  Positive" class="gt_row gt_right">8</td>
<td headers="Jun  Total" class="gt_row gt_right">25</td>
<td headers="Jun  Positive_Percentage" class="gt_row gt_right" style="background-color: #FF0000; color: #FFFFFF;">32%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Jul">Jul</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Jul  Negative" class="gt_row gt_right">8</td>
<td headers="Jul  Positive" class="gt_row gt_right">17</td>
<td headers="Jul  Total" class="gt_row gt_right">25</td>
<td headers="Jul  Positive_Percentage" class="gt_row gt_right" style="background-color: #ECFF00; color: #000000;">68%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Aug">Aug</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Aug  Negative" class="gt_row gt_right">11</td>
<td headers="Aug  Positive" class="gt_row gt_right">14</td>
<td headers="Aug  Total" class="gt_row gt_right">25</td>
<td headers="Aug  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFD100; color: #000000;">56%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Sep">Sep</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Sep  Negative" class="gt_row gt_right">9</td>
<td headers="Sep  Positive" class="gt_row gt_right">16</td>
<td headers="Sep  Total" class="gt_row gt_right">25</td>
<td headers="Sep  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFFF00; color: #000000;">64%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Oct">Oct</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Oct  Negative" class="gt_row gt_right">10</td>
<td headers="Oct  Positive" class="gt_row gt_right">15</td>
<td headers="Oct  Total" class="gt_row gt_right">25</td>
<td headers="Oct  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFE800; color: #000000;">60%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Nov">Nov</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Nov  Negative" class="gt_row gt_right">1</td>
<td headers="Nov  Positive" class="gt_row gt_right">24</td>
<td headers="Nov  Total" class="gt_row gt_right">25</td>
<td headers="Nov  Positive_Percentage" class="gt_row gt_right" style="background-color: #00FF00; color: #000000;">96%</td></tr>
    <tr class="gt_group_heading_row">
      <th colspan="4" class="gt_group_heading" scope="colgroup" id="Dec">Dec</th>
    </tr>
    <tr class="gt_row_group_first"><td headers="Dec  Negative" class="gt_row gt_right">12</td>
<td headers="Dec  Positive" class="gt_row gt_right">13</td>
<td headers="Dec  Total" class="gt_row gt_right">25</td>
<td headers="Dec  Positive_Percentage" class="gt_row gt_right" style="background-color: #FFBA00; color: #000000;">52%</td></tr>
  </tbody>
  
  
</table>
</div>
```
:::
:::

## Remarks

November has the highest percentage of positive months (96%), making it
the most consistently strong month. June has the lowest percentage of
positive months (32%), indicating it is the weakest. April (76%), March
(69%), and July (68%) show strong positive trends, while January (48%),
May (52%), and December (52%) are more balanced. This data highlights
seasonal trends in historical market performance.
