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
            Log.d(TAG, "Données GPS reçues dans Fragment2: '$gpsData'")
            val parts = gpsData.split(",")
            if (parts.size >= 3) {
                val lat = parts[0]
                val lon = parts[1]
                val alt = parts[2]
                gpsTextView.text = "Localisation GPS\n\nLatitude: $lat\nLongitude: $lon\nAltitude: $alt"
            } else {
                gpsTextView.text = "Localisation GPS\n\nDonnées reçues:\n$gpsData"
            }
            Log.d(TAG, "TextView GPS mis à jour")
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