package com.example.APPLI

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView

private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

class Fragment2 : Fragment() {
    private val TAG = "Fragment2"
    private var param1: String? = null
    private var param2: String? = null

    private lateinit var gpsTextView: TextView
    private val sharedDataManager: SharedDataManager by activityViewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_2, container, false)
        
        gpsTextView = view.findViewById(R.id.textView2)
        gpsTextView.text = "Localisation GPS\n\nEn attente de données..."
        
        sharedDataManager.gpsData.observe(viewLifecycleOwner) { gpsData ->
            Log.d(TAG, "Données GPS reçues pour traitement dans Fragment2: '$gpsData'")

            // Regex pour trouver tous les nombres (entiers ou décimaux, y compris négatifs)
            val numberPattern = Regex("-?\\d+\\.?\\d*")
            val numbers = numberPattern.findAll(gpsData).map { it.value }.toList()

            if (numbers.size >= 3) {
                // Assez de nombres pour lat, lon, et alt
                val lat = numbers[0]
                val lon = numbers[1]
                val alt = numbers[2]
                gpsTextView.text = "Localisation GPS\n\nLatitude: $lat\nLongitude: $lon\nAltitude: $alt"
            } else if (numbers.size == 2) {
                // Assez pour lat et lon
                val lat = numbers[0]
                val lon = numbers[1]
                gpsTextView.text = "Localisation GPS\n\nLatitude: $lat\nLongitude: $lon"
            }
            // Si moins de 2 nombres sont trouvés, on ne met pas à jour l'affichage
            // pour éviter d'afficher des données incomplètes.
        }
        
        Log.d(TAG, "Fragment2 onCreateView terminé")
        return view
    }

    companion object {
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            Fragment2().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}