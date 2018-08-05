import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_rma_data import load_rma_loss_ratio_cause, load_rma_loss_all

# Bar chart for the top 10 causes of loss from the RMA loss data, including indemnity amount, count, and area

data_loss = load_rma_loss_all(crop_name='corn')

sns.set_context("notebook")
sns.set_style("ticks")

bin_cause_amount = data_loss.groupby('Damage Cause Description').sum()['Indemnity Amount'].sort_values(ascending=False)[0:10]
bin_cause_count = data_loss.groupby('Damage Cause Description').count()['Indemnity Amount'].sort_values(ascending=False)[0:10]
bin_cause_area = data_loss.groupby('Damage Cause Description').sum()['Determined Acres'].sort_values(ascending=False)[0:10]

fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(12,5))
(bin_cause_amount/1e9).plot.bar(ax=ax1, width=0.5)
ax1.set_title('The total amount of loss')
ax1.set_ylabel('Indemnity amount (billion US dolar)')

ax1.text(-0.12, 1.05, 'a', fontsize=16, transform=ax1.transAxes, fontweight='bold')

(bin_cause_count/1e3).plot.bar(ax=ax2, width=0.5)
ax2.set_title('The total count')
ax2.set_ylabel('Count (thousand)')
ax2.text(-0.12, 1.05, 'b', fontsize=16, transform=ax2.transAxes, fontweight='bold')


(bin_cause_area/1e6).plot.bar(ax=ax3, width=0.5)
ax3.set_title('Acres lost due to damage (2001-2016)')
ax3.set_ylabel('Determined acers (million)')
ax3.text(-0.12, 1.05, 'c', fontsize=16, transform=ax3.transAxes, fontweight='bold')


sns.despine()

plt.subplots_adjust(wspace=0.15, bottom=0.4)
plt.savefig('../figure/figure_bar_cause_indemnity_count_area.pdf')
