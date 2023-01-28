# Hopfield
> 類神經網路的 project 3

---

* 選擇訓練資料、測試資料、epoch、是否同步後，即可開始用 Hopfield 網路進行訓練

* 訓練完成後，可看到在設定的 epoch 數內，每個 epoch 的訓練結果

* 訓練過程
    * 將圖形平坦化成一維資料
    * $N$ 為訓練資料數量，$p$ 為圖形大小，以下為每一個 epoch 會做的事：
        * $W = \frac{1}{p} \sum\limits_{k=1}^N {x_k x_k^T} - \frac{N}{p} I$
        * $\theta_j = \sum\limits_{i=1}^p w_{ji}$
        * $y = \sum\limits_{i=1}^p W_{ji} x_j(n) - \theta_j$
            * $y > 0 \rightarrow x_j(n+1) = 1$
            * $y = 0 \rightarrow x_j(n+1) = x_j(n)$
            * $y < 0 \rightarrow x_j(n+1) = -1$
        * 非同步為一次調整一列之 $x$，同步則是一起計算再調整

* 附有許多範例訓練、測試資料，可參考格式並自訂訓練、測試資料

* 缺點：本作法為純 Hopfield，沒有任何額外的機制，因此重疊部分過多的圖形將容易聯想失敗
