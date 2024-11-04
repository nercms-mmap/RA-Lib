# RA-Lib
RA-Lib is a benchmarking environment for Rank Aggregation (RA) algorithms. This website contains the current benchmarking results, which have 21 unsupervised RA methods, 7 supervised RA methods and 1 semi-supervised RA methods, these algorithms were tested on our preprocessed datasets. These datasets cover the areas of person re-identification (re-ID), recommendation system, bioinformatics and social choice. The code of tested methods includes both classical and state-of-the-art RA methods that can be funded in https://github.com/nercms-mmap. As well as having all the experimental details and settings on this website. 

If you want to add your own algorithm to improve the benchmarking system, please send a package of your algorithm code and a link to the published paper to waii2022@whu.edu.cn.


<p align="center">
  <img src="https://github.com/user-attachments/assets/2484bc74-4b5c-4d51-96c2-1e16d3941025" width="1000">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Comb*-NIST SP'94-143240" alt="" />  
  <img src="https://img.shields.io/badge/MC1--4-WWW'01-1E4A5E" alt="" /> 
  <img src="https://img.shields.io/badge/wBorada-WWW'12-1E4A5E" alt="" />  
  <img src="https://img.shields.io/badge/BordaCount-SIGIR'01-27627D" alt="" />  
  <img src="https://img.shields.io/badge/Dowdall-IPSR'02-317A9B" alt="" />  
  <img src="https://img.shields.io/badge/Median-SIMOD'03-53A4C9" alt="" />  
  <img src="https://img.shields.io/badge/RRF-SIMOD'09-53A4C9" alt="" />  
  <img src="https://img.shields.io/badge/iRANK-JIST'10-71B5D3" alt="" /> 
  <img src="https://img.shields.io/badge/Mean-PMLR'11-90C4DC" alt="" />  
  <img src="https://img.shields.io/badge/HPA-ECIR'20-AED5E6" alt="" />  
  <img src="https://img.shields.io/badge/PostNDCG-ECIR'20-AED5E6" alt="" />  
  <img src="https://img.shields.io/badge/ER-OMEGA'20-CCE4EF" alt="" />  
  <img src="https://img.shields.io/badge/Mork--H-EJOR'20-F5DFD8" alt="" />  
  <img src="https://img.shields.io/badge/CG-JORS'21-EDC5B9" alt="" />  
  <img src="https://img.shields.io/badge/DIBRA-LSA'22-E5AB99" alt="" />  
  <img src="https://img.shields.io/badge/SSRA-CIKM'08-C69191" alt="" />
  <img src="https://img.shields.io/badge/CRF-CIKM'13-C69191" alt="" />  
  <img src="https://img.shields.io/badge/CSRA-ICASSP'20-B97777" alt="" />  
  <img src="https://img.shields.io/badge/AggRankDE-Electronics'22-AB5D5D" alt="" />  
  <img src="https://img.shields.io/badge/IRA-BMVC'22-C25759" alt="" />  
  <img src="https://img.shields.io/badge/Borda--Score-AAAI'23-B14144" alt="" />  
  <img src="https://img.shields.io/badge/QI--IRA-AAAI'24-B14144" alt="" />  
</p>


<table align="center">
    <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Unsupervised</b>
      </td>
      <td>
        <b>Supervised</b>
      </td>
      <td>
        <b>Semi-supervised</b>
      </td>
    </tr>
    <tr valign="top">
        <td>
            <ul>
                <li>$\textrm{CombMIN}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombMAX}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombSUM}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombANZ}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombMNZ}$ <a href="#Comb">[1]</li>
                <li>$\textrm{MC1}$ <a href="#MC">[2]</li>
                <li>$\textrm{MC2}$ <a href="#MC">[2]</li>
                <li>$\textrm{MC3}$ <a href="#MC">[2]</li>
                <li>$\textrm{MC4}$ <a href="#MC">[2]</li>
                <li>$\textrm{Borda count}$ <a href="#Borda">[3]</li>
                <li>$\textrm{Dowdall}$ <a href="#Dowdall">[4]</li>
                <li>$\textrm{Median}$ <a href="#Median">[5]</li>
                <li>$\textrm{RRF}$ <a href="#RRF">[6]</li>
                <li>$\textrm{iRANK}$ <a href="#iRANK">[7]</li>
                <li>$\textrm{Mean}$ <a href="#Mean">[8]</li>
                <li>$\textrm{HPA}$ <a href="#HPA&postNDCG">[9]</li>
                <li>$\textrm{PostNDCG}$ <a href="#HPA&postNDCG">[9]</li>
                <li>$\textrm{ER}$ <a href="#ER">[10]</li>
                <li>$\textrm{Mork-H}$ <a href="#Mork-H">[11]</li>
                <li>$\textrm{CG}$ <a href="#CG">[12]</li>
                <li>$\textrm{DIBRA}$ <a href="#DIBRA">[13]</li>
        </td>
        <td>
            <ul>
                <li>$\textrm{wBorda}$ <a href="#wBorda">[14]</li>
                <li>$\textrm{CRF}$ <a href="#CRF">[15]</li>
                <li>$\textrm{CSRA}$ <a href="#CSRA">[16]</li>
                <li>$\textrm{AggRankDE}$ <a href="#AggRankDe">[17]</li>
                <li>$\textrm{IRA}_\textrm{R}$ <a href="#IRA">[18]</li>
                <li>$\textrm{IRA}_\textrm{S}$ <a href="#IRA">[18]</li>
                <li>$\textrm{QI-IRA}$ <a href="#QIIRA">[19]</li>
        </td>
        <td>
            <ul>
                <li>$\textrm{SSRA}$ <a href="#semi">[20]</li>
        </td>
    </tbody>
</table>

# Directory Structure
```
│  README.md
│  plot.py
| 
├─results
│  ├─re-ID.csv
│  ├─social choice.csv
│      
```

Demo
=======


CUHK03 (labeled)
-----------------
![image](https://github.com/user-attachments/assets/847600af-8c25-47ad-8151-3a603803a056)

Running
=======

1. Run `python plot.py` to plot results.

Follow-up Plan
=======
We will be updating and adding more RA methods for shared use.

References
=======
<a id="Comb">[[1]](https://books.google.com.tw/books?hl=zh-CN&lr=&id=W8MZAQAAIAAJ&oi=fnd&pg=PA243&dq=Combination+of+multiple+searches.&ots=3XwVWFAQ5n&sig=EGO4Nkeo5BIsfg0HOpiHsnNPjm4&redir_esc=y#v=onepage&q=Combination%20of%20multiple%20searches.&f=false) Fox, E., & Shaw, J. (1994). Combination of multiple searches. NIST special publication SP, 243-243.</a>

<a id="MC">[[2]](https://dl.acm.org/doi/abs/10.1145/371920.372165) Dwork, C., Kumar, R., Naor, M., & Sivakumar, D. (2001, April). Rank aggregation methods for the web. In Proceedings of the 10th international conference on World Wide Web (pp. 613-622).</a>

<a id="Borda">[[3]](https://dl.acm.org/doi/abs/10.1145/383952.384007) Aslam, J. A., & Montague, M. (2001, September). Models for metasearch. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 276-284).</a>

<a id="Dowdall">[[4]](https://journals.sagepub.com/doi/abs/10.1177/0192512102023004002) Reilly, B. (2002). Social choice in the south seas: Electoral innovation and the borda count in the pacific island countries. International Political Science Review, 23(4), 355-372.</a>

<a id="Median">[[5]](https://dl.acm.org/doi/abs/10.1145/872757.872795) Fagin, R., Kumar, R., & Sivakumar, D. (2003, June). Efficient similarity search and classification via rank aggregation. In Proceedings of the 2003 ACM SIGMOD international conference on Management of data (pp. 301-312).</a>

<a id="RRF">[[6]](https://dl.acm.org/doi/abs/10.1145/1571941.1572114) Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009, July). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).</a>

<a id="iRANK">[[7]](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.21296) Wei, F., Li, W., & Liu, S. (2010). iRANK: A rank‐learn‐combine framework for unsupervised ensemble ranking. Journal of the American Society for Information Science and Technology, 61(6), 1232-1243.</a>

<a id="Mean">[[8]](https://proceedings.mlr.press/v14/burges11a/burges11a.pdf) Burges, C., Svore, K., Bennett, P., Pastusiak, A., & Wu, Q. (2011, January). Learning to rank using an ensemble of lambda-gradient models. In Proceedings of the learning to rank Challenge (pp. 25-35). PMLR.</a>

<a id="HPA&postNDCG">[[9]](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_17) Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised Ensemble of Ranking Models for News Comments Using Pseudo Answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.</a>

<a id="ER">[[10]](https://www.sciencedirect.com/science/article/pii/S0305048319308448) Mohammadi, M., & Rezaei, J. (2020). Ensemble ranking: Aggregation of rankings produced by different multi-criteria decision-making methods. Omega, 96, 102254.</a>

<a id="Mork-H">[[11]](https://www.sciencedirect.com/science/article/abs/pii/S0377221719307039) Ivano Azzini., & Giuseppe Munda. (2020). Azzini, I., & Munda, G. (2020). A new approach for identifying the Kemeny median ranking. European Journal of Operational Research, 281(2), 388-401. </a>

<a id="CG">[[12]](https://www.tandfonline.com/doi/abs/10.1080/01605682.2019.1657365) Xiao, Y., Deng, H. Z., Lu, X., & Wu, J. (2021). Graph-based rank aggregation method for high-dimensional and partial rankings. Journal of the Operational Research Society, 72(1), 227-236.</a>

<a id="DIBRA">[[13]](https://www.sciencedirect.com/science/article/abs/pii/S0957417422007710) Akritidis, L., Fevgas, A., Bozanis, P., & Manolopoulos, Y. (2022). An unsupervised distance-based model for weighted rank aggregation with list pruning. Expert Systems with Applications, 202, 117435.</a>

<a id="wBorda">[[14]](https://ieeexplore.ieee.org/abstract/document/6495123) Pujari, M., & Kanawati, R. (2012, November). Link prediction in complex networks by supervised rank aggregation. In 2012 IEEE 24th International Conference on Tools with Artificial Intelligence (Vol. 1, pp. 782-789). IEEE.</a>

<a id="CRF">[[15]](https://www.jmlr.org/papers/volume15/volkovs14a/volkovs14a.pdf) Volkovs, M. N., & Zemel, R. S. (2014). New learning methods for supervised and unsupervised preference aggregation. The Journal of Machine Learning Research, 15(1), 1135-1176.</a>

<a id="CSRA">[[16]](https://ieeexplore.ieee.org/abstract/document/9053496) Yu, Y., Liang, C., Ruan, W., & Jiang, L. (2020, May). Crowdsourcing-Based Ranking Aggregation for Person Re-Identification. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1933-1937). IEEE.</a>

<a id="AggRankDe">[[17]](https://www.mdpi.com/2079-9292/11/3/369) Bałchanowski, M., & Boryczka, U. (2022). Aggregation of Rankings Using Metaheuristics in Recommendation Systems. Electronics, 11(3), 369.</a>

<a id="IRA">[[18]](https://bmvc2022.mpi-inf.mpg.de/0386.pdf) Huang, J., Liang, C., Zhang, Y., Wang, Z., & Zhang, C. (2022). Ranking Aggregation with Interactive Feedback for Collaborative Person Re-identification.</a>

<a id="QIIRA">[[19]](https://aaai.org/wp-content/uploads/2024/01/AAAI_Main-Track_2024-01-04.pdf) Hu, C., Zhang, H., Liang, C., & Huang, H. (2024). QI-IRA: Quantum-inspired interactive ranking aggregation for person re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, pp. 1-9).</a>

<a id="semi">[[20]](https://dl.acm.org/doi/abs/10.1145/1458082.1458315) Chen, S., Wang, F., Song, Y., & Zhang, C. (2008, October). Semi-supervised ranking aggregation. In Proceedings of the 17th ACM conference on Information and knowledge management (pp. 1427-1428).</a>

 ## Contacts

 If you encounter any problems, you can contact us via email 2021302111226@whu.edu.cn
