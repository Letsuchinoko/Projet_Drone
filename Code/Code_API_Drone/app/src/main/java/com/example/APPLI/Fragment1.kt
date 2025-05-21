package com.example.APPLI

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment

class Fragment1 : Fragment() {
    private lateinit var textView: TextView
    private lateinit var btnPermissionsBT: Button
    private lateinit var btnActiverBT: Button
    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var bluetoothAdapter: BluetoothAdapter

    // Effectue la demande des autorisations nécessaires
    // Si tout est OK, cela activera le bouton « Activer BT »
    private val requetePermissions: ActivityResultLauncher<Array<String>> =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (permissions.values.all { it }) {
                btnActiverBT.isEnabled = true
                Toast.makeText(activity, "Permissions OK", Toast.LENGTH_SHORT).show()
            }
        }

    // Gère la demande d'activation du Bluetooth
    private val requeteActivationBT =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                Toast.makeText(activity, "L'utilisateur a activé le BT",
                    Toast.LENGTH_SHORT).show()
                // Activer un bouton (A COMPLETER PLUS TARD)
            } else {
                Toast.makeText(activity, "L'utilisateur a refusé",
                    Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_1, container, false)

        // Initialisation des vues
        textView = view.findViewById(R.id.textView)
        btnPermissionsBT = view.findViewById(R.id.btnPermissionsBT)
        btnActiverBT = view.findViewById(R.id.btnActiverBT)

        // Configuration initiale
        textView.text = "En attente des permissions Bluetooth..."
        btnActiverBT.isEnabled = false

        // Gestion des clics sur les boutons avec Toast
        btnPermissionsBT.setOnClickListener {
            lateinit var BT_PERMISSIONS: Array<String>
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                BT_PERMISSIONS = arrayOf(
                    Manifest.permission.BLUETOOTH_CONNECT,
                    Manifest.permission.BLUETOOTH_SCAN
                )
            } else {
                BT_PERMISSIONS = arrayOf(
                    Manifest.permission.BLUETOOTH_ADMIN,
                    Manifest.permission.BLUETOOTH,
                    Manifest.permission.ACCESS_FINE_LOCATION
                )
            }
            requetePermissions.launch(BT_PERMISSIONS)
        }

        btnActiverBT.setOnClickListener {
            if (bluetoothAdapter.isEnabled) {
                Toast.makeText(activity, "BT déjà activé", Toast.LENGTH_SHORT).show()
                // Activer un bouton (A COMPLETER PLUS TARD)
            } else {
                val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
                requeteActivationBT.launch(enableBtIntent)
            }
        }

        // Tester si l'appareil ciblé autorise l'interface BT
        bluetoothManager = requireContext().getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter
        if (bluetoothAdapter == null) {
            // L'appareil n'est pas doté d'une interface BT
            Toast.makeText(activity, "La machine ne possède pas le Bluetooth",
                Toast.LENGTH_SHORT).show()
        } else {
            btnPermissionsBT.isEnabled = true
            Toast.makeText(activity, "Interface BT existe", Toast.LENGTH_SHORT).show()
        }

        return view
    }
}